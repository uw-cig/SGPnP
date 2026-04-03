# param.py
import numpy as np
params = {
    "save_dir": 'experiment',                       # Experiment saving directory
    "inverse_problem_type": 'super_resolution',     # ['box_inpainting', 'random_inpainting', 'blur', 'super_resolution]
    "pretrained_model_path": '/project/cigserver3/export1/p.youngil/pretrained_models/Diffusion_Model/dps/ffhq_10m.pt',
    "acceleration_rate": 4, 
    "measurement_noise_level": 0.02, 
    "hyperparam1": 1.48,                            # Data consistency step size
    "hyperparam2": 0.95,                            # Prior step size
    "pnp_method": "pgm",
    "kernel_index": 0, 
    "pnp_iters": 100, 
    "sigma_cond_0": 20.,                            # for classical PnP methods like DPIR, IRCNN
    "sigma_cond_K": 0.02,                           # for classical PnP methods like DPIR, IRCNN
    "sigma_inject_0": 0.,                           # for classical PnP methods like DPIR, IRCNN
    "sigma_inject_K": 0.,                           # for classical PnP methods like DPIR, IRCNN
    "dataset_name": 'ffhq',
    "quick_validation": '2', 
    "mask_pattern": 'randomly_cartesian',
    "method": 'diffusion',
}

def get_params():
    return params