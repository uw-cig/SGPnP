# param.py
# param.py
import numpy as np

params = {
    "save_dir": 'experiment',                       # Experiment saving directory
    "inverse_problem_type": 'blur',     # ['box_inpainting', 'random_inpainting', 'blur', 'super_resolution]
    "pretrained_model_path": 'pretrained_models/ffhq_10m.pt',
    "acceleration_rate": 4, 
    "measurement_noise_level": 0.02, 
    "hyperparam1": 1.55,                            # Data consistency step size
    "hyperparam2": 0.,                            # Prior step size
    "pnp_method": "sgpnp_dpir",
    "kernel_index": 0, 
    "pnp_iters": 200, 
    "sigma_cond_0": 25.,                            # for classical PnP methods like DPIR, IRCNN
    "sigma_cond_K": 0.02,                           # for classical PnP methods like DPIR, IRCNN
    "sigma_inject_0": 25.,                           # for classical PnP methods like DPIR, IRCNN
    "sigma_inject_K": 0.02,                           # for classical PnP methods like DPIR, IRCNN
    "dataset_name": 'ffhq',
    "quick_validation": '2', 
    "mask_pattern": 'randomly_cartesian',
    "method": 'diffusion',
}

def get_params():
    return params