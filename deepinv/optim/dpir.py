from deepinv.optim import BaseOptim
from deepinv.models import DRUNet
from deepinv.optim.data_fidelity import L2
from deepinv.optim.prior import PnP
from deepinv.optim.optimizers import create_iterator
import numpy as np


def get_DPIR_params(lamb, max_noise_level, min_noise_level, max_iter = 200):
    r"""
    Default parameters for the DPIR Plug-and-Play algorithm.

    :param float noise_level_img: Noise level of the input image.
    """
    s1 = max_noise_level

    s2 = min_noise_level

    sigma_denoiser = np.logspace(np.log10(s1), np.log10(s2), max_iter).astype(
        np.float32
    )
    stepsize = (sigma_denoiser / max(0.01, min_noise_level)) ** 2
    
    return list(sigma_denoiser), list(lamb * stepsize), max_iter

def get_SGPnP_params(lamb, sigma_cond_0, sigma_cond_K, sigma_inject_0, sigma_inject_K, max_iter = 200):
    r"""
    Default parameters for the DPIR Plug-and-Play algorithm.

    :param float noise_level_img: Noise level of the input image.
    """
    sigma_cond_list = np.logspace(np.log10(sigma_cond_0), np.log10(sigma_cond_K), max_iter).astype(np.float32)
    sigma_inject_list = np.logspace(np.log10(sigma_inject_0), np.log10(sigma_inject_K), max_iter).astype(np.float32)
    stepsize = (sigma_cond_list / max(0.01, sigma_cond_K)) ** 2
    
    return list(sigma_cond_list), list(sigma_inject_list), list(lamb * stepsize)

def get_sigma_inject_params(max_noise_level, max_iter, min_noise_level = 0):
    r"""
    Default parameters for the DPIR Plug-and-Play algorithm.

    :param float noise_level_img: Noise level of the input image.
    """
    s1 = max_noise_level

    s2 = min_noise_level# * S2_noise_level_scaling_factor

    sigma_denoiser = np.logspace(np.log10(s1), np.log10(s2), max_iter).astype(
        np.float32
    )
    
    return list(sigma_denoiser)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_DPIR_params_for_generation(noise_level_img, lamb, max_noise_level, min_noise_level, max_iter = 200):
    r"""
    Default parameters for the DPIR Plug-and-Play algorithm.

    :param float noise_level_img: Noise level of the input image.
    """
    start = 0
    end = 3
    # tau = 0.3
    tau = lamb
    clip_min = 1e-9

    s1 = max_noise_level

    s2 = noise_level_img# * S2_noise_level_scaling_factor
    t = np.linspace(0, 1, max_iter)

    v_start = sigmoid(start / tau)
    v_end = sigmoid(end / tau)

    output = sigmoid((t * (end - start) + start) / tau)

    output = (v_end - output) / (v_end - v_start)
    output = np.clip(output, clip_min, 1.0)

    sigma = output * (s1 - s2) + s2
    sigma.astype(np.float32)

    sigma_denoiser = sigma
    stepsize = (sigma_denoiser / max(0.01, noise_level_img)) ** 2

    print(f"lamb: {lamb} / sigma_denoiser: {sigma_denoiser}\n s1: {s1} / s2: {s2} / max_iter: {max_iter}")
    return list(sigma_denoiser), list(lamb * stepsize), max_iter



class DPIR(BaseOptim):
    r"""
    Deep Plug-and-Play (DPIR) algorithm for image restoration.

    The method is based on half-quadratic splitting (HQS) and a PnP prior with a pretrained denoiser :class:`deepinv.models.DRUNet`.
    The optimization is stopped early and the noise level for the denoiser is adapted at each iteration.
    See :ref:`sphx_glr_auto_examples_plug-and-play_demo_PnP_DPIR_deblur.py` for more details on the implementation,
    and how to adapt it to your specific problem.

    This method uses a standard :math:`\ell_2` data fidelity term.

    The DPIR method is described in Zhang, K., Zuo, W., Gu, S., & Zhang, L. (2017). "Learning deep CNN denoiser prior for image restoration"
    In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 3929-3938).

    :param float sigma: Standard deviation of the measurement noise, which controls the choice of the
        rest of the hyperparameters of the algorithm. Default is ``0.1``.
    """

    def __init__(self, sigma=0.1, device="cuda"):
        prior = PnP(denoiser=DRUNet(pretrained="download", device=device))
        sigma_denoiser, stepsize, max_iter = get_DPIR_params(sigma)
        params_algo = {"stepsize": stepsize, "g_param": sigma_denoiser}
        super(DPIR, self).__init__(
            create_iterator("HQS", prior=prior, F_fn=None, g_first=False),
            max_iter=max_iter,
            data_fidelity=L2(),
            prior=prior,
            early_stop=False,
            params_algo=params_algo,
        )
