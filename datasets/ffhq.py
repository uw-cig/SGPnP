from glob import glob
from PIL import Image
from typing import Callable, Optional
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import random



__DATASET__ = {}

def save_individual_image(image, title, save_path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image, cmap='gray')
    ax.set_title(title)
    ax.axis('off')
    # Make directory first
    # check_and_mkdir(save_path)
    plt.savefig(save_path)
    plt.close(fig)

def register_dataset(name: str):
    def wrapper(cls):
        if __DATASET__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __DATASET__[name] = cls
        return cls
    return wrapper

def random_box(b, h, w, degradation_ratio):
    if h == 256:
        patch_size = 32
    elif h == 128:
        patch_size = 16
    elif h == 64:
        patch_size = 8
    else:
        raise ValueError(f"Unknown image size {h}.")
    coarse_mask = (torch.rand((b, 1, h // patch_size, w // patch_size)) > degradation_ratio).float()

    mask = F.interpolate(coarse_mask, size=(h, w), mode='bicubic', align_corners=False)

    mask = (mask > 0.5).int()

    return mask

import torch.nn.functional as F
import glob

@register_dataset(name='lsunbedroom')
class LSUNBedroomDataset(VisionDataset):
    def __init__(self, diffusion_model_type, root: str, mode: str = "train", transforms: Optional[Callable]=None):
        super().__init__(root, transforms)
        
        all_fpaths = glob.glob(os.path.join(root, "*.webp"))
        assert len(all_fpaths) > 0, "File list is empty. Check the root path and file extension."
        
        self.fpaths = all_fpaths  # Use 1000–70000 for training

        self.mode = mode  # Store mode for debugging or future extensions
        self.diffusion_model_type = diffusion_model_type
        self.random_flip = False

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        
        fpath = self.fpaths[index]
        x = Image.open(fpath).convert('RGB')
                
        if self.transforms is not None:
            x = self.transforms(x)
        else:
            raise ValueError("Not implemented yet.")

        x = x.unsqueeze(0)
        
        return {'x': x, 'smps': x}

@register_dataset(name='ffhq')
class FFHQDataset(VisionDataset):
    def __init__(self, diffusion_model_type, root: str, mode: str = "train", transforms: Optional[Callable]=None):
        super().__init__(root, transforms)
        
        # all_fpaths = [f"{root}/{str(i).zfill(5)}.png" for i in range(70000)]
        all_fpaths = [os.path.join(root, f"{i:05d}.png") for i in range(70000)]
        assert len(all_fpaths) > 0, "File list is empty. Check the root."
        
        if mode == "train":
            self.fpaths = all_fpaths[1000:]  # Use 1000–70000 for training
        elif mode == "valid":
            self.fpaths = all_fpaths[500:1000]  # Use 500–1000 for validation
        elif mode == "test":
            # self.fpaths = all_fpaths[:1000]  # Use 0–1000 for testing
            self.fpaths = all_fpaths[:500]  # Use 0–500 for testing
        else:
            raise ValueError(f"Unknown mode {mode}. Use 'train' or 'test'.")

        self.mode = mode  # Store mode for debugging or future extensions
        self.diffusion_model_type = diffusion_model_type
        self.random_flip = False

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        
        fpath = self.fpaths[index]
        x = Image.open(fpath).convert('RGB')
        
        if self.transforms is not None:
            x = self.transforms(x)
        else:
            raise ValueError("Not implemented yet.")

        x = x.unsqueeze(0)
        b, c, h, w = x.shape
        
        return {'x': x, 'smps': x}
        
def get_ffhqdataset(name: str, root: str, mode: str, diffusion_model_type: str, **kwargs):
    if __DATASET__.get(name, None) is None:
        raise NameError(f"Dataset {name} is not defined.")
    # Pass mode to the dataset initialization
    return __DATASET__[name](root=root, mode=mode, diffusion_model_type = diffusion_model_type, **kwargs)

def get_ffhqdataloader(dataset: VisionDataset,
                   batch_size: int, 
                   num_workers: int, 
                   train: bool):
    dataloader = DataLoader(dataset, 
                            batch_size, 
                            shuffle=train, 
                            num_workers=num_workers, 
                            drop_last=train)
    return dataloader

def get_ffhq_mask(batch_size, image_w_h, mask_pattern, dataset_name, acceleration_rate):
    # raise ValueError("")
    assert (mask_pattern in ['randomly_cartesian', 'uniformly_cartesian', 'mix_cartesian', 'random_box']) and (dataset_name in ['ffhq', 'fastmri']) 
    if mask_pattern == "randomly_cartesian" and dataset_name == "fastmri":
        mask_np = _mask_fn[mask_pattern](img_size = (image_w_h, image_w_h), acceleration_rate = acceleration_rate)
        mask = np.expand_dims(mask_np, 0)  # add batch dimension
        mask = (torch.from_numpy(mask).to(torch.float32)).unsqueeze(1)
    elif mask_pattern == "random_box" and dataset_name == "ffhq":
        mask = random_box(b = batch_size, h = image_w_h, w = image_w_h, degradation_ratio = acceleration_rate)
    else:
        raise ValueError(f"Not yet to be implemented mask_pattern: {mask_pattern}")
    
    return mask

def get_square_mask(batch_size, img, image_w_h, acceleration_rate):
    """
    image_size = image_w_h
    l, h = [128, 129]
    l, h = int(l), int(h)
    mask_h = np.random.randint(l, h)
    mask_w = np.random.randint(l, h)
    mask_shape = (mask_h, mask_w)

    margin = (16, 16)
    
    B, C, H, W = img.shape
    h, w = mask_shape
    margin_height, margin_width = margin
    maxt = image_size - margin_height - h
    maxl = image_size - margin_width - w

    # bb
    t = np.random.randint(margin_height, maxt)
    l = np.random.randint(margin_width, maxl)

    # make mask
    mask = torch.ones([B, C, H, W], device=img.device)
    mask[..., t:t+h, l:l+w] = 0

    print(f"mask.shape: {mask.shape}")
    ----------------------------
    img.shape: torch.Size([1, 3, 256, 256])
    mask_shape: (128, 128)
    image_size: 256
    margin: (16, 16)
    ----------------------------

    """
    image_size = image_w_h
    B, C, H, W = img.shape
    # l, h = [128, 128]
    # l, h = int(l), int(h)
    # mask_h = np.random.randint(l, h)
    # mask_w = np.random.randint(l, h)
    mask_shape = (128, 128)
    h, w = mask_shape
    margin = (16, 16)
    
    print(f"*********************")
    print(f"img.shape: {img.shape}")
    print(f"mask_shape: {mask_shape}")
    print(f"image_size: {image_size}")
    print(f"margin: {margin}")
    print(f"*********************")

    margin_height, margin_width = margin
    maxt = image_size - margin_height - h
    maxl = image_size - margin_width - w

    # bb
    t = np.random.randint(margin_height, maxt)
    l = np.random.randint(margin_width, maxl)

    # make mask
    mask = torch.ones([B, C, H, W], device=img.device)
    mask[..., t:t+h, l:l+w] = 0

    return mask#, t, t+h, l, l+w