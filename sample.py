# -----------------
# Importing from Python module
# -----------------
import torch
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import tifffile as tiff
import deepinv
from pathlib import Path

# -----------------
# Importing from files
# -----------------
from guided_diffusion import logger
from guided_diffusion.script_util import sr_model_and_diffusion_defaults, dicts_to_dict, sr_create_model_and_diffusion
from utility.file_utility import mkdir_exp_recording_folder, load_yaml
from utility.img_utils import mask_generator
from datasets.fastMRI import fastMRI, ftran, get_fastmri_mask
from datasets.ffhq import get_ffhqdataset
from utility.metric_utility import compute_psnr_ssim_nmse_lpips, normalize_np
from utility.sgp_utility import load_param_module, create_argparser, to_numpy_list


def main():
    # -----------------
    # 1. Get the input arguments
    # -----------------
    parser = create_argparser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument('--config', type=str)
    parser.add_argument('--save_tiff', type=str, default="false")
    args = parser.parse_args()
    torch.cuda.device_count()
    param_module = load_param_module(args.config)
    param_dicts = param_module.get_params()
    gpu = args.gpu

    # -----------------
    # 2. Extract information from configurations
    # -----------------
    dataset_name = param_dicts['dataset_name'].lower()
    if dataset_name == "ffhq":
        diffusion_config = load_yaml('configs/diffusion_config/ffhq_diffusion_config.yaml')
    elif dataset_name == "fastmri":
        diffusion_config = load_yaml('configs/diffusion_config/fastmri_diffusion_config.yaml')
    else:
        raise ValueError(f"Check the dataset_name: {dataset_name}")
    save_tiff = True if (args.save_tiff).lower() == "true" else False; save_dir = param_dicts['save_dir']; mask_pattern = param_dicts['mask_pattern'].lower(); acceleration_rate = param_dicts['acceleration_rate']; hyperparam1 = param_dicts['hyperparam1']; hyperparam2 = param_dicts['hyperparam2']; pnp_method = param_dicts['pnp_method']; kernel_index = param_dicts['kernel_index']; pnp_iters = param_dicts['pnp_iters']; sigma_cond_0 = param_dicts['sigma_cond_0']; sigma_cond_K = param_dicts['sigma_cond_K']; sigma_inject_0 = param_dicts['sigma_inject_0']; sigma_inject_K = param_dicts['sigma_inject_K'];inverse_problem_type = param_dicts['inverse_problem_type']; inverse_problem_config = diffusion_config['inverse_problems'][inverse_problem_type]; measurement_noise_level = param_dicts['measurement_noise_level']; diffusion_model_type  = diffusion_config['diffusion']['diffusion_model_type']; param_dicts['diffusion_model_type'] = diffusion_model_type; image_size = diffusion_config['diffusion']['image_size']; learn_sigma = diffusion_config['diffusion']['learn_sigma']; in_channels = diffusion_config['diffusion']['in_channels']; cond_channels = diffusion_config['diffusion']['cond_channels']; num_channels = diffusion_config['diffusion']['num_channels']; num_res_blocks = diffusion_config['diffusion']['num_res_blocks']; channel_mult = diffusion_config['diffusion']['channel_mult']; class_cond = diffusion_config['diffusion']['class_cond']; use_checkpoint = diffusion_config['diffusion']['use_checkpoint']; attention_resolutions = diffusion_config['diffusion']['attention_resolutions']; num_heads = diffusion_config['diffusion']['num_heads']; num_head_channels = diffusion_config['diffusion']['num_head_channels']; num_heads_upsample = diffusion_config['diffusion']['num_heads_upsample']; use_scale_shift_norm = diffusion_config['diffusion']['use_scale_shift_norm']; dropout = diffusion_config['diffusion']['dropout']; resblock_updown = diffusion_config['diffusion']['resblock_updown']; use_fp16 = diffusion_config['diffusion']['use_fp16']; use_new_attention_order = diffusion_config['diffusion']['use_new_attention_order']; diffusion_predtype = diffusion_config['diffusion']['diffusion_predtype']; pretrained_model_path = param_dicts['pretrained_model_path']; predict_xstart = True if diffusion_predtype == "pred_xstart" else False; clip_denoised = diffusion_config['test']['diffusion']['clip_denoised'];use_ddim = diffusion_config['test']['diffusion']['use_ddim'];timestep_respacing = diffusion_config['test']['diffusion']['timestep_respacing'];device_str = f"cuda:{gpu}" if torch.cuda.is_available() else 'cpu'; device = torch.device(device_str);num_steps = 1000 if timestep_respacing == "" else int(timestep_respacing[4:]);param_dicts['num_steps'] = num_steps; predict_xstart = True if diffusion_predtype == "pred_xstart" else False; quick_validation = int(param_dicts['quick_validation']) if param_dicts['quick_validation'] != '' else -1
    save_dir, _ = mkdir_exp_recording_folder(save_dir=save_dir, dataset_name=dataset_name, image_size=image_size, ffhq_batch_size = 1, fastmri_batch_size = 1, lidcidri_batch_size = 1, weight_decay = None, dropout = None)
    os.environ['OPENAI_LOGDIR'] = str(save_dir) # set the logdir
    logger.configure()
    logger.log("creating model and diffusion...")
    logger.log("# ------------\n Creating data loader...\n# ------------\n")

    # -----------------
    # 3. Load dataset
    # -----------------
    if dataset_name == "fastmri":
        latent_shape = (1, 2, image_size, image_size)
        dataset = fastMRI(mode='test', mask_pattern=mask_pattern, root = "demo_data/fastmri", image_size = image_size, diffusion_model_type=diffusion_model_type)
        dataloader = DataLoader(dataset,
                        batch_size=1,
                        num_workers=1,
                        shuffle=False,
                        drop_last=False)
    elif dataset_name in ["ffhq"]:
        latent_shape = (1, 3, image_size, image_size)
        transform = transforms.Compose([
                                transforms.Resize((image_size, image_size)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])
        dataset = get_ffhqdataset(mode = "test", name = dataset_name, root = "demo_data/ffhq", transforms=transform,
                                diffusion_model_type = diffusion_model_type)
        dataloader = DataLoader(dataset,
                        batch_size=1,
                        num_workers=1,
                        shuffle=False,
                        drop_last=False)
    else:
        raise ValueError("Check the dataset_name")
    


    # -----------------
    # 4. Load pretrained model
    # -----------------
    model_diffusion_dict = {
        'image_size': image_size,'large_size': image_size,'small_size': image_size,'num_channels': num_channels,'num_res_blocks': num_res_blocks,'num_heads': num_heads,'num_heads_upsample': num_heads_upsample,'num_head_channels': num_head_channels,'attention_resolutions': attention_resolutions,'channel_mult': channel_mult,'dropout': dropout,'class_cond': class_cond,'use_checkpoint': use_checkpoint,'use_scale_shift_norm': use_scale_shift_norm,'resblock_updown': resblock_updown,'use_fp16': use_fp16,'use_new_attention_order':use_new_attention_order,'learn_sigma': learn_sigma,'diffusion_steps': 1000,'noise_schedule': 'linear','timestep_respacing': timestep_respacing,'use_kl': False,'predict_xstart': predict_xstart,'rescale_timesteps': True if use_ddim else False,'rescale_learned_sigmas': False,'in_channels': in_channels,'cond_channels': cond_channels,'diffusion_model_type': diffusion_model_type,'for_training': True,
    }
    print(f"# --------------\npretrained_model path: {pretrained_model_path} loaded...\n# --------------\n")
    pretrained_model, diffusion = sr_create_model_and_diffusion(
        **dicts_to_dict(model_diffusion_dict, sr_model_and_diffusion_defaults().keys()), param_dicts = param_dicts
    )
    pretrained_model.load_state_dict(torch.load(pretrained_model_path, map_location="cpu"))
    pretrained_model.to(device)
    pretrained_model.eval()
    
    if param_dicts['pnp_method'] in ["sgpnp_pgm", "sgpnp_dpir", "sgpnp_admm"]:
        sample_fn = (
            diffusion.SGPnP_Iterations
        )
    elif param_dicts['pnp_method'] in ["pgm", "dpir", "pnpadmm"]:
        sample_fn = (
            diffusion.SDPnP_Iterations
        )
    else:
        raise ValueError(f"param_dicts['pnp_method'] {param_dicts['pnp_method']}")
    
    logger.log("# ------------\n Sampling...\n# ------------\n")

    inputs = []
    recons = []
    gts = []

    input_psnr_list = []; recon_psnr_list = []; input_ssim_list = []; recon_ssim_list = []; input_nmse_list = []; recon_nmse_list = []; input_lpips_list = []; recon_lpips_list = []

    # -----------------
    # 5. Load data to solve inverse problems
    # -----------------
    for data_index, data_dicts in enumerate(dataloader):
        x_gt, smps = data_dicts['x'], data_dicts['smps']
        
        x_gt = (x_gt.squeeze(1)).to(device) 
        smps = (smps.squeeze(1)).to(device) 

        model_kwargs = {}

        if dataset_name in ["ffhq"]:
            if inverse_problem_type in ['box_inpainting', 'random_inpainting']:
                mask_gen = mask_generator(
                    **inverse_problem_config['mask_opt']
                    )
                
                mask = mask_gen(x_gt)

                mask = mask[:, 0, :, :].unsqueeze(dim=1)
                
                physics = deepinv.physics.Inpainting(
                    mask=mask, tensor_size=latent_shape[1:],
                    noise_model=deepinv.physics.GaussianNoise(measurement_noise_level),
                    device=device
                )
                
                model_kwargs['low_res'] = mask.to(device)
                model_kwargs['smps'] = mask.to(device)
                y_hat = physics(x_gt)

                y_hat_image = y_hat

            elif inverse_problem_type in ['blur']:
                from deepinv.utils.demo import load_degradation
                kernel_dir = "demo_data/kernel"
                DEBLUR_KER_DIR = Path(kernel_dir)
                kernel_torch = load_degradation(name="Levin09.npy", data_dir = DEBLUR_KER_DIR, index=kernel_index, download=False)
                kernel_torch = kernel_torch.unsqueeze(0).unsqueeze(0)  # add batch and channel dimensions
                kernel_torch = kernel_torch.to(torch.float32)

                physics = deepinv.physics.BlurFFT(
                    img_size=latent_shape[1:],
                    filter=kernel_torch,
                    device=device,
                    noise_model=deepinv.physics.GaussianNoise(sigma=measurement_noise_level),
                )
                y_hat = physics(x_gt)
                model_kwargs['low_res'] = y_hat
                model_kwargs['smps'] = y_hat
                
                
                y_hat_image = y_hat

            elif inverse_problem_type in ['super_resolution']:
                physics = deepinv.physics.Downsampling(
                        img_size = latent_shape[1:],
                        filter='bicubic',
                        factor=acceleration_rate,
                        device=x_gt.device,
                        padding="circular",
                        noise_model=deepinv.physics.GaussianNoise(sigma=measurement_noise_level),
                )
                y_hat = physics(x_gt)
                y_hat_image = physics.A_adjoint(y_hat)
                model_kwargs['low_res'] = y_hat
                model_kwargs['smps'] = y_hat
                
            else:
                raise ValueError(f"Check the inverse_problem_type {inverse_problem_type}")

        elif dataset_name == "fastmri":
            b,c,h,w = x_gt.shape
            
            if inverse_problem_type in ['fastmri_reconstruction']:
                mask, mask_np = get_fastmri_mask(batch_size = b, image_w_h = h, mask_pattern = mask_pattern, acceleration_rate = acceleration_rate, dataset_name = "fastmri")
                mask = mask.to(device)
                smps = smps.to(device) 
                model_kwargs['low_res'] = mask
                model_kwargs['smps'] = smps

                physics = deepinv.physics.MultiCoilMRI(
                        img_size = x_gt.shape[2:],
                        device=x_gt.device,
                        noise_model=deepinv.physics.GaussianNoise(sigma=measurement_noise_level),
                        mask = mask,
                        coil_maps = smps,
                )
                
                y_hat = physics(x_gt)
                
                y_hat_image = physics.A_adjoint(y_hat)

        else:
            raise ValueError(f"Check the dataset_name: {dataset_name}")
        
        # -----------------
        # 6. Run PnP iterations
        # -----------------
        sample = sample_fn(
            model=pretrained_model,
            shape = latent_shape,
            model_kwargs=model_kwargs,
            param_dicts = param_dicts,
            measurement = y_hat,
            physics = physics,
        )
        
        # -----------------
        # 7. Evaluate performance
        # -----------------

        if dataset_name in ["ffhq"]:
            if inverse_problem_type in ['super_resolution']:
                y_hat = y_hat_image

            y_hat_np = normalize_np(y_hat.squeeze().detach().cpu().numpy().transpose(1, 2, 0))
            sample_np = normalize_np(sample.squeeze().detach().cpu().numpy().transpose(1, 2, 0))
            x_gt_np = normalize_np(x_gt.squeeze().detach().cpu().numpy().transpose(1, 2, 0))

            input_img = torch.tensor(y_hat_np).to(device)
            recon_img = torch.tensor(sample_np).to(device)
            recon_gt = torch.tensor(x_gt_np).to(device)

            input_img_cal = input_img# * mask_roi
            recon_img_cal = recon_img# * mask_roi

        elif dataset_name == "fastmri":
            input_img = torch.abs(torch.view_as_complex(ftran(y_hat, mask = model_kwargs['low_res'], smps = model_kwargs['smps'])[0].permute(1,2,0).contiguous().cpu().detach()))
            input_img = (input_img - input_img.min()) / (input_img.max() - input_img.min())

            recon_img = torch.abs(torch.view_as_complex(sample[0].permute(1,2,0).contiguous().cpu().detach()))
            recon_img = (recon_img - recon_img.min()) / (recon_img.max() - recon_img.min())

            recon_gt = torch.abs(torch.view_as_complex(x_gt[0].permute(1, 2, 0).contiguous().cpu().detach()))
            recon_gt = (recon_gt - recon_gt.min()) / (recon_gt.max() - recon_gt.min())

            mask_roi = recon_gt.clone()
            mask_roi[mask_roi != 0] = 1

            input_img_cal = input_img# * mask_roi
            recon_img_cal = recon_img# * mask_roi
        
        else:
            raise ValueError(f"Check the dataset_name: {dataset_name}")
        
        print(f"[{data_index+1}/{len(dataloader)}] {inverse_problem_type} / {acceleration_rate} / {measurement_noise_level} / lambda {hyperparam1} / zeta {hyperparam2} / pnp_method {pnp_method} / sigma_cond_0: {sigma_cond_0} / sigma_cond_K: {sigma_cond_K} / sigma_inject_0: {sigma_inject_0} / sigma_inject_K: {sigma_inject_K} / pnp_iters: {pnp_iters}")
        
        if inverse_problem_type in ['ffhq_generation', 'fastmri_generation', 'lidcidri_generation']:
            input_psnr_value, input_ssim_value, input_nmse_value, input_lpips_value = torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0)
            recon_psnr_value, recon_ssim_value, recon_nmse_value, recon_lpips_value = torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0)

        else:
            input_psnr_value, input_ssim_value, input_nmse_value, input_lpips_value = compute_psnr_ssim_nmse_lpips(input_img_cal, recon_gt, device = device)
            recon_psnr_value, recon_ssim_value, recon_nmse_value, recon_lpips_value = compute_psnr_ssim_nmse_lpips(recon_img_cal, recon_gt, device = device)
        

        print(f"[{data_index+1}/{len(dataloader)}] {inverse_problem_type} / {acceleration_rate} / {measurement_noise_level} / lambda {hyperparam1} / zeta {hyperparam2}")
        print(f"input_psnr_value: {input_psnr_value} / input_ssim_value: {input_ssim_value} input_lpips_value: {input_lpips_value}\nrecon_psnr_value: {recon_psnr_value} / recon_ssim_value: {recon_ssim_value} / recon_lpips_value: {recon_lpips_value}")
        
        
        inputs.append(input_img_cal.unsqueeze(0))
        recons.append(recon_img_cal.unsqueeze(0))
        gts.append(recon_gt.unsqueeze(0))
        input_psnr_list.append(input_psnr_value.item())
        recon_psnr_list.append(recon_psnr_value)
        input_ssim_list.append(input_ssim_value)
        recon_ssim_list.append(recon_ssim_value)
        input_nmse_list.append(input_nmse_value)
        recon_nmse_list.append(recon_nmse_value)
        input_lpips_list.append(input_lpips_value)
        recon_lpips_list.append(recon_lpips_value)
        
        if data_index == quick_validation:
            print(f'----------------- Quick Validation -----------------')
            break

    avg_input_psnr_value = np.mean(to_numpy_list(input_psnr_list))
    avg_input_ssim_value = np.mean(to_numpy_list(input_ssim_list))
    avg_input_lpips_value = np.mean(to_numpy_list(input_lpips_list))
    std_input_psnr_value = np.std(to_numpy_list(input_psnr_list))
    std_input_ssim_value = np.std(to_numpy_list(input_ssim_list))
    std_input_lpips_value = np.std(to_numpy_list(input_lpips_list))
    avg_recon_psnr_value = np.mean(to_numpy_list(recon_psnr_list))
    avg_recon_ssim_value = np.mean(to_numpy_list(recon_ssim_list))
    avg_recon_lpips_value = np.mean(to_numpy_list(recon_lpips_list))
    std_recon_psnr_value = np.std(to_numpy_list(recon_psnr_list))
    std_recon_ssim_value = np.std(to_numpy_list(recon_ssim_list))
    std_recon_lpips_value = np.std(to_numpy_list(recon_lpips_list))


    print('----------------- Summary -----------------')
    print(f"{inverse_problem_type} / {acceleration_rate} / {measurement_noise_level} / CG_iters {hyperparam1} / gamma {hyperparam2} / pnp_method {pnp_method}")
    print('----------------- Total Average -----------------')
    print(f"avg input psnr: {avg_input_psnr_value} / avg input ssim: {avg_input_ssim_value} avg input lpips: {avg_input_lpips_value}\navg recon psnr: {avg_recon_psnr_value} / avg recon ssim: {avg_recon_ssim_value} / avg recon lpips: {avg_recon_lpips_value}")
    print('----------------- Total STD -----------------')
    print(f"std input psnr: {std_input_psnr_value} / std input ssim: {std_input_ssim_value} std input lpips: {std_input_lpips_value}\nstd recon psnr: {std_recon_psnr_value} / std recon ssim: {std_recon_ssim_value} / std recon lpips: {std_recon_lpips_value}")
    

    if save_tiff == True:
        recon_arr = torch.cat(recons, axis=0)
        recon_arr *= 255
        recon_arr = recon_arr.to("cpu", torch.uint8).numpy()

        input_arr = torch.cat(inputs, axis=0)
        input_arr *= 255
        input_arr = input_arr.to("cpu", torch.uint8).numpy()

        gt_arr = torch.cat(gts, axis=0)
        gt_arr *= 255
        gt_arr = gt_arr.to("cpu", torch.uint8).numpy()
        
            
        tiff.imwrite(os.path.join(save_dir, f"input_acc_{inverse_problem_type}_{acceleration_rate}_sigma{measurement_noise_level}.tiff"), input_arr, imagej=True)
        tiff.imwrite(os.path.join(save_dir, f"recon_{inverse_problem_type}_acc{acceleration_rate}_sigma{measurement_noise_level}_numsteps{num_steps}_hyper1{hyperparam1}_hyper2{hyperparam2}_pnpmethod{pnp_method}_iters{pnp_iters}_sigmacond0{sigma_cond_0}_sigmainject0{sigma_inject_0}.tiff"), recon_arr, imagej=True)
        tiff.imwrite(os.path.join(save_dir, f"gt.tiff"), gt_arr, imagej=True)

    logger.log("sampling complete")

if __name__ == "__main__":
    main()
