"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""


# -----------------
# Importing from Python module
# -----------------
import enum
import math
import numpy as np
import torch as th
import torch
import tqdm

# -----------------
# Importing from files
# -----------------
from datasets.fastMRI import ftran
from deepinv.optim.dpir import get_DPIR_params, get_SGPnP_params
from deepinv.optim.data_fidelity import L2


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps, diffusion_model_type):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        scale = 1000 / num_diffusion_timesteps
    
        beta_start = scale * 0.0001
            
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon

class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()

class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL

class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()
        self.T = 1000

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)
        
        self.diffusion_sigma_t = self.sqrt_one_minus_alphas_cumprod / self.sqrt_alphas_cumprod

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )
    
    def get_alphas(self, num_train_timesteps, device, beta_start=0.1 / 1000, beta_end=20 / 1000):
        betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
        betas = torch.from_numpy(betas).to(device)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas.cpu(), axis=0)
        return alphas_cumprod.clone().detach()
        # return torch.tensor(alphas_cumprod)

    def get_alpha_beta(self, num_train_timesteps, device, beta_start = 0.1 / 1000, beta_end = 20 / 1000):
        """
        Get the alpha and beta sequences for the algorithm. This is necessary for mapping noise levels to timesteps.
        """
        betas = np.linspace(
            beta_start, beta_end, num_train_timesteps, dtype=np.float32
        )
        betas = torch.from_numpy(betas).to(device)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas.cpu(), axis=0)  # This is \overline{\alpha}_t

        # Useful sequences deriving from alphas_cumprod
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_1m_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        reduced_alpha_cumprod = torch.div(
            sqrt_1m_alphas_cumprod, sqrt_alphas_cumprod
        )  # equivalent noise sigma on image
        sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1)

        return (
            sqrt_1m_alphas_cumprod,
            reduced_alpha_cumprod,
            sqrt_alphas_cumprod,
            sqrt_recip_alphas_cumprod,
            sqrt_recipm1_alphas_cumprod,
            betas,
        )
    
    def sigma_timestep_matching(self, sigmas, noise_level_list):
        interpolation_number = 1000000
        extended_sigmas = np.interp(np.linspace(0, 1, interpolation_number), np.linspace(0, 1, len(sigmas)), sigmas)
        
        # Find the closest indices for each noise level
        indices = [np.abs(extended_sigmas - nl).argmin() for nl in noise_level_list]
        indexed_extended_sigmas = extended_sigmas[indices]
        pnp_alphas_list = 1/((np.array(indexed_extended_sigmas**2) + 1))
        pnp_alphas_list = pnp_alphas_list.tolist()
        pnp_timesteps_list = ((np.array(indices) / interpolation_number)) * 999
        pnp_timesteps_list = pnp_timesteps_list.tolist()
        
        return pnp_timesteps_list, pnp_alphas_list

    def SGPnP_Iterations(self, model, shape, model_kwargs, param_dicts, measurement, physics):
        # -----------------
        # 1. Define necessary variables and data fidelity function
        # -----------------
        dataset_name = param_dicts['dataset_name']; dc_stepsize = param_dicts['hyperparam1']; prior_stepsize = param_dicts['hyperparam2']; pnp_method = param_dicts['pnp_method']; num_steps = param_dicts['num_steps']; pnp_iters = param_dicts['pnp_iters']; sigma_cond_0 = param_dicts['sigma_cond_0']; sigma_cond_K = param_dicts['sigma_cond_K']; sigma_inject_0 = param_dicts['sigma_inject_0']; sigma_inject_K = param_dicts['sigma_inject_K']; mask = model_kwargs['low_res']; mps = model_kwargs['smps']; device = measurement.device
        assert pnp_method in ["sgpnp_pgm", "sgpnp_dpir", "sgpnp_admm"]
        if param_dicts["inverse_problem_type"] in ["super_resolution"]:
            measurement_micro_image = physics.A_adjoint(measurement)
        else:
            measurement_micro_image = measurement
        if dataset_name in ['fastmri']:
            mps = (model_kwargs['smps'].squeeze(1))#.contiguous()
            mps = mps.permute(0, 3, 1, 2).contiguous()
            measurement_copy = torch.view_as_complex(measurement.permute([0, 4, 2, 3, 1]).contiguous())
            measurement_micro_image = ftran(measurement_copy, smps = model_kwargs['smps'], mask = mask)
            measurement_micro_image = (torch.view_as_real(measurement_micro_image)).permute([0, 3, 1, 2]).contiguous()
        data_fidelity = L2()

        # -----------------
        # 2. Define alphas and sigmas for noise-to-timestep matching
        # -----------------
        alphas = self.get_alphas(num_train_timesteps=self.T, device = device)
        sigmas = torch.sqrt(1.0 - alphas) / alphas.sqrt()
        sigma_cond_list, sigma_inject_list, stepsize = get_SGPnP_params(lamb=dc_stepsize, sigma_cond_0 = sigma_cond_0, sigma_cond_K = sigma_cond_K, sigma_inject_0 = sigma_inject_0, sigma_inject_K = sigma_inject_K, max_iter = pnp_iters)
        pnp_timesteps_list, pnp_alphas_list = self.sigma_timestep_matching(sigmas = sigmas, noise_level_list = sigma_cond_list)
        timesteps_alphas_pair = zip(pnp_timesteps_list, pnp_alphas_list)
        assert len(pnp_timesteps_list) == len(pnp_alphas_list)
        
        # -----------------
        # 3. Initialize variables for PnP iterations
        # -----------------
        xk = measurement_micro_image

        if pnp_method in ["sgpnp_admm"]:
            zk = measurement_micro_image.clone().detach()
            sk = torch.zeros_like(xk)  

        loop_index = 0
        # -----------------
        # 4. Run PnP iterations
        # -----------------
        for indexed_t, indexed_alphas in tqdm.tqdm(timesteps_alphas_pair, total=len(pnp_timesteps_list)):
            t = torch.tensor([indexed_t] * shape[0], device=device, dtype=torch.long); at = torch.tensor(indexed_alphas, device=device)
            
            if pnp_method == "sgpnp_pgm":
                # -----------------
                # 4-1. Data consistency step
                # -----------------
                dc_pred_xstart = data_fidelity.prox(xk, measurement, physics, gamma = stepsize[loop_index])
                # -----------------
                # 4-2. Noise injection step
                # -----------------
                noise = th.randn_like(measurement_micro_image)
                noise_level = sigma_inject_list[loop_index]
                noisy_xk = at.sqrt() * (xk + noise_level * noise)
                # -----------------
                # 4-3. Denoising step using diffusion model
                # -----------------
                with torch.no_grad():
                    et = model(noisy_xk, self._scale_timesteps(t))
                    if et.size(1) == shape[1]*2:
                        et = et[:, :et.size(1) // 2]
                    else:
                        pass
                x0_t = (noisy_xk - et * (1 - at).sqrt()) / at.sqrt()
                # -----------------
                # 4-4. Update
                # -----------------

                residual = xk - x0_t

                xk = dc_pred_xstart - prior_stepsize * residual
            
            elif pnp_method == "sgpnp_admm":
                # -----------------
                # 4-1. Data consistency step
                # -----------------
                xk = data_fidelity.prox(zk - sk, measurement, physics, gamma = stepsize[loop_index])
                # -----------------
                # 4-2. Noise injection step
                # -----------------
                noise = th.randn_like(measurement_micro_image)
                xk_sk = xk + sk
                noise_level = sigma_inject_list[loop_index]
                noisy_xk_sk = at.sqrt() * (xk_sk + noise_level * noise)
                # -----------------
                # 4-3. Denoising step using diffusion model
                # -----------------
                with torch.no_grad():
                    et = model(noisy_xk_sk, self._scale_timesteps(t))
                    if et.size(1) == shape[1]*2:
                        et = et[:, :et.size(1) // 2]
                    else:
                        pass
                zk = (noisy_xk_sk - et * (1 - at).sqrt()) / at.sqrt()

                # -----------------
                # 4-4. Update
                # -----------------
                sk = sk + xk - zk
                
            elif pnp_method == "sgpnp_dpir":
                # -----------------
                # 4-1. Data consistency step
                # -----------------
                dc_pred_xstart = data_fidelity.prox(xk, measurement, physics, gamma = stepsize[loop_index])
                # -----------------
                # 4-2. Noise injection step
                # -----------------
                noise = th.randn_like(measurement_micro_image)
                noise_level = sigma_inject_list[loop_index]
                noisy_xk = at.sqrt() * (dc_pred_xstart + noise_level * noise)
                # -----------------
                # 4-3. Denoising step using diffusion model
                # -----------------
                with torch.no_grad():
                    et = model(noisy_xk, self._scale_timesteps(t))
                    if et.size(1) == shape[1]*2:
                        et = et[:, :et.size(1) // 2]
                    else:
                        pass
                x0_t = (noisy_xk - et * (1 - at).sqrt()) / at.sqrt()

                # -----------------
                # 4-4. Update
                # -----------------
                xk = x0_t
                    
            else:
                raise ValueError(f"Check the method: {pnp_method}")
        
            loop_index += 1

        if pnp_method == "sgpnp_admm":
            x = zk
        else:
            x = xk
        
        return x

    def SDPnP_Iterations(self, model, shape, model_kwargs, param_dicts, measurement, physics):
        # -----------------
        # 1. Define necessary variables and data fidelity function
        # -----------------
        dataset_name = param_dicts['dataset_name']; dc_stepsize = param_dicts['hyperparam1']; prior_stepsize = param_dicts['hyperparam2']; pnp_method = param_dicts['pnp_method']; num_steps = param_dicts['num_steps'];sigma_cond_0 = param_dicts['sigma_cond_0']; sigma_cond_K = param_dicts['sigma_cond_K'];mask = model_kwargs['low_res'];mps = model_kwargs['smps'];pnp_iters = param_dicts['pnp_iters'];b, c, h, w = shape;device = measurement.device
        if param_dicts["inverse_problem_type"] in ["super_resolution"]:
            measurement_micro_image = physics.A_adjoint(measurement)
        else:
            measurement_micro_image = measurement
        if dataset_name in ['fastmri']:
            mps = (model_kwargs['smps'].squeeze(1))#.contiguous()
            mps = mps.permute(0, 3, 1, 2).contiguous()
            measurement_copy = torch.view_as_complex(measurement.permute([0, 4, 2, 3, 1]).contiguous())
            measurement_micro_image = ftran(measurement_copy, smps = model_kwargs['smps'], mask = mask)
            measurement_micro_image = (torch.view_as_real(measurement_micro_image)).permute([0, 3, 1, 2]).contiguous()
        data_fidelity = L2()
        
        # -----------------
        # 2. Define alphas and sigmas for noise-to-timestep matching
        # -----------------
        alphas = self.get_alphas(num_train_timesteps=self.T, device = device)
        sigmas = torch.sqrt(1.0 - alphas) / alphas.sqrt()
        sigma_cond_list, stepsize, max_iter = get_DPIR_params(lamb=dc_stepsize, max_noise_level = sigma_cond_0, min_noise_level = sigma_cond_K, max_iter = pnp_iters)
        pnp_timesteps_list, pnp_alphas_list = self.sigma_timestep_matching(sigmas = sigmas, noise_level_list = sigma_cond_list)
        timesteps_alphas_pair = zip(pnp_timesteps_list, pnp_alphas_list)
        assert len(pnp_timesteps_list) == len(pnp_alphas_list)

        # -----------------
        # 3. Initialize variables for PnP iterations
        # -----------------
        xk = measurement_micro_image

        if pnp_method in ["pnpadmm"]:
            zk = measurement_micro_image.clone().detach()
            sk = torch.zeros_like(xk)
        
        loop_index = 0

        # -----------------
        # 4. Run PnP iterations
        # -----------------
        for indexed_t, indexed_alphas in tqdm.tqdm(timesteps_alphas_pair, total=len(pnp_timesteps_list)):
            # -----------------
            # 4-0. Necessary scaling term for denoising using diffusion model
            # -----------------
            c = math.sqrt(indexed_alphas)
            indexed_t_tensor = torch.tensor([indexed_t] * shape[0], device=device)

            if pnp_method == "pgm":
                # -----------------
                # 4-1. Data consistency step
                # -----------------
                dc_pred_xstart = data_fidelity.prox(xk, measurement, physics, gamma = stepsize[loop_index])

                # -----------------
                # 4-2. Denoising step using end-to-end model or diffusion model
                # -----------------
                if param_dicts["method"] == "end2end":
                    model_input = xk
                    model_kwargs['noise_level'] = th.tensor([sigma_cond_list[loop_index]], device=model_input.device)
                    with torch.no_grad():
                        x0_t = model(model_input, self._scale_timesteps(indexed_t_tensor), **model_kwargs)

                elif param_dicts["method"] == "diffusion":
                    model_input = c * xk
                    with torch.no_grad():
                        et = model(model_input, self._scale_timesteps(indexed_t_tensor))
                    if et.size(1) == shape[1]*2:
                        et = et[:, :et.size(1) // 2]
                    else:
                        pass
                    x0_t = (model_input - et * math.sqrt(1 - indexed_alphas)) / math.sqrt(indexed_alphas)
                    
                else:
                    raise ValueError(f"Unknown method: {param_dicts['method']}")
                
                # -----------------
                # 4-3. Update
                # -----------------
                    
                residual = xk - x0_t
                xk = dc_pred_xstart - prior_stepsize * residual


            elif pnp_method == "pnpadmm":
                # -----------------
                # 4-1. Data consistency step
                # -----------------
                xk = data_fidelity.prox(zk - sk, measurement, physics, gamma = stepsize[loop_index])

                # -----------------
                # 4-2. Denoising step using end-to-end model or diffusion model
                # -----------------
                if param_dicts["method"] == "end2end":
                    model_input = xk + sk
                    model_kwargs['noise_level'] = th.tensor([sigma_cond_list[loop_index]], device=model_input.device)
                    with torch.no_grad():
                        zk = model(model_input, self._scale_timesteps(indexed_t_tensor), **model_kwargs)

                elif param_dicts["method"] == "diffusion":
                    model_input = c * (xk + sk)
                    with torch.no_grad():
                        et = model(model_input, self._scale_timesteps(indexed_t_tensor))
                        if et.size(1) == shape[1]*2:
                            et = et[:, :et.size(1) // 2]
                        else:
                            pass
                    zk = (model_input - et * math.sqrt(1 - indexed_alphas)) / math.sqrt(indexed_alphas)           

                else:
                    raise ValueError(f"Unknown method: {param_dicts['method']}")

                # -----------------
                # 4-3. Update
                # -----------------
                    
                sk = sk + xk - zk                

            elif pnp_method == "dpir":
                # -----------------
                # 4-1. Data consistency step
                # -----------------
                dc_pred_xstart = data_fidelity.prox(xk, measurement, physics, gamma = stepsize[loop_index])

                # -----------------
                # 4-2. Denoising step using end-to-end model or diffusion model
                # -----------------
                if param_dicts["method"] == "end2end":
                    model_input = dc_pred_xstart
                    model_kwargs['noise_level'] = th.tensor([sigma_cond_list[loop_index]], device=model_input.device)
                    with torch.no_grad():
                        x0_t = model(model_input, self._scale_timesteps(indexed_t_tensor), **model_kwargs)
                    
                elif param_dicts["method"] == "diffusion":
                    model_input = c * (dc_pred_xstart)
                    with torch.no_grad():
                        et = model(model_input, self._scale_timesteps(indexed_t_tensor))
                        if et.size(1) == shape[1]*2:
                            et = et[:, :et.size(1) // 2]
                        else:
                            pass
                    x0_t = (model_input - et * math.sqrt(1 - indexed_alphas)) / math.sqrt(indexed_alphas)
                else:
                    raise ValueError(f"Unknown method: {param_dicts['method']}")
                # -----------------
                # 4-3. Update
                # -----------------
                xk = x0_t

            else:
                raise ValueError(f"Unknown PnP method: {pnp_method}")
            
            loop_index += 1

        if pnp_method == "pnpadmm":
            x = zk
        else:
            x = xk
        
        return x