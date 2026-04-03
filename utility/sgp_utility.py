from guided_diffusion.script_util import model_and_diffusion_defaults, add_dict_to_argparser
import importlib
import argparse
import torch

def load_param_module(param_file):
    spec = importlib.util.spec_from_file_location("param_module", param_file)
    param_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(param_module)
    return param_module

def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def to_numpy_list(lst):
    return [t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t for t in lst]

