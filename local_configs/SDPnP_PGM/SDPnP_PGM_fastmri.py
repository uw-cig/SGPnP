# param.py
# param.py
import numpy as np

params = {
    "save_dir": 'experiment',                       # Experiment saving directory
    "inverse_problem_type": 'fastmri_reconstruction',     # ['fastmri_reconstruction']
    "pretrained_model_path": '/project/cigserver3/export1/p.youngil/pretrained_models/Diffusion_Model/2025_MDIFF/fmri_oracle_EPSmatchPred/01232025_0229_fastmri_vbF_randmask_acc1_img256_batch1_noiseindata_0.0_tauSure_0_lrSure_0_tvFalse_lrTv0/ema_0.9999_2200000.pt',
    "acceleration_rate": 4, 
    "measurement_noise_level": 0.02, 
    "hyperparam1": 1.4,                            # Data consistency step size
    "hyperparam2": 0.63,                            # Prior step size
    "pnp_method": "pgm",
    "kernel_index": 0, 
    "pnp_iters": 200, 
    "sigma_cond_0": 7.5,                            # for classical PnP methods like DPIR, IRCNN
    "sigma_cond_K": 0.02,                           # for classical PnP methods like DPIR, IRCNN
    "sigma_inject_0": 0.,                           # for classical PnP methods like DPIR, IRCNN
    "sigma_inject_K": 0.,                           # for classical PnP methods like DPIR, IRCNN
    "dataset_name": 'fastmri',
    "quick_validation": '2', 
    "mask_pattern": 'randomly_cartesian',
    "method": 'diffusion',
}

def get_params():
    return params