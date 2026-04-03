import torch
import os
import time
from datetime import datetime
from pathlib import Path
import json
import math
import matplotlib.pyplot as plt
import yaml


def mkdir_exp_recording_folder(save_dir, dataset_name, image_size, ffhq_batch_size, fastmri_batch_size, lidcidri_batch_size, dropout, weight_decay, lr = None, experiment_type = None):
    """
    measurement_operator_name example: inpainting
    """
    current_time = time.time()
    current_date = datetime.now().strftime("%m%d%Y")
    current_hour_minute = datetime.now().strftime("%H%M")
        
    unique_name = f"{current_date}_{current_hour_minute}_{dataset_name}_{experiment_type}_img{image_size}_lr{lr}_ffhqb{ffhq_batch_size}_fastmrib{fastmri_batch_size}_lidcidrib{lidcidri_batch_size}_do_{dropout}_wd_{weight_decay}"

    result_file = Path(save_dir) / unique_name / "results.csv"
    os.makedirs(Path(save_dir) / unique_name, exist_ok=True)
    result_dir = Path(save_dir) / unique_name
    return result_dir, result_file


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def check_and_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)