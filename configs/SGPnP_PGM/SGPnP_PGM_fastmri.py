# param.py
# param.py
import numpy as np

params = {
    "save_dir": 'experiment',                       # Experiment saving directory
    "inverse_problem_type": 'fastmri_reconstruction',     # ['fastmri_reconstruction']
    "pretrained_model_path": 'pretrained_models/fmri_uncond_R1_noise0_0_img256.pt',
    "acceleration_rate": 4, 
    "measurement_noise_level": 0.02, 
    "hyperparam1": 1.7,                            # Data consistency step size
    "hyperparam2": 0.75,                            # Prior step size
    "pnp_method": "sgpnp_pgm",
    "kernel_index": 0, 
    "pnp_iters": 200, 
    "sigma_cond_0": 1.,                            # for classical PnP methods like DPIR, IRCNN
    "sigma_cond_K": 0.02,                           # for classical PnP methods like DPIR, IRCNN
    "sigma_inject_0": 0.01,                           # for classical PnP methods like DPIR, IRCNN
    "sigma_inject_K": 0.001,                           # for classical PnP methods like DPIR, IRCNN
    "dataset_name": 'fastmri',
    "quick_validation": '2', 
    "mask_pattern": 'randomly_cartesian',
    "method": 'diffusion',
}

def get_params():
    return params