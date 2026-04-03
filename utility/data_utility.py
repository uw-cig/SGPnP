import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

def make_angles(num_angles, shift=0.0, device="cpu"):
    """
    Generate uniformly spaced tomography angles in [0, 180),
    but shifted by an offset.

    Args:
        num_angles (int): number of angles
        shift (float): starting offset in degrees
        device: torch device
    """
    step = 180.0 / num_angles
    # generate shifted grid, wrap around 180 if needed
    angles = (torch.arange(num_angles, device=device) * step + shift) % 180.0
    return angles


def renorm_from_minusonetoone_to_zeroone(x):
    return (x + 1.) / 2.

class DatasetWrapper(Dataset):
    def __init__(self, parent_dataset, diffusion_model_type, dataset_name):
        self.dataset = parent_dataset
        self.dataset_name = dataset_name
        self.diffusion_model_type = diffusion_model_type

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        # print(f"item: {item}")
        
        if self.dataset_name == "fastmri":
            x = self.dataset[item]['x']
            smps = self.dataset[item]['smps']
        elif self.dataset_name == "lidcidri":
            x = self.dataset[item]['x']
            smps = self.dataset[item]['smps']
        elif self.dataset_name in ["ffhq", "imagenet", "lsunbedroom"]:
            x = self.dataset[item]['x']
            smps = self.dataset[item]['smps']
            # raise ValueError("not yet implemented")
        else:
            raise ValueError(f"Check the dataset_name {self.dataset_name}")

        return {'x': x, 'smps': smps}

def training_dataloader_wrapper(dataset, batch_size, num_workers):
    # raise ValueError(f"dataset: {dataset} \n batch_size: {batch_size} \n num_workers: {num_workers}")
    data_loader = DataLoader(
        dataset=dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    while True:
        yield from data_loader
        
def testing_dataloader_wrapper(dataset, batch_size, num_workers):
    # raise ValueError(f"dataset: {dataset} \n batch_size: {batch_size} \n num_workers: {num_workers}")
    data_loader = DataLoader(
        dataset=dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    
    # return data_loader

    while True:
        yield from data_loader


def crop_images(images, crop_width):
    # raise ValueError(f"images.shape: {images.shape}")
    if len(images.shape) != 4:
        raise ValueError("Check the size of the images")
    # crop_width = min(images.shape[2], images.shape[3])

    cropped_images = []

    for i in range(images.shape[0]):  # Iterate over the number of images
        img = images[i]  # Select one image
        for j in range(images.shape[1]):  # Iterate over the channels
            channel = img[j]  # Select one channel
            cropping_transform = transforms.CenterCrop((crop_width, crop_width))
            cropped_channel = cropping_transform(channel)
            cropped_images.append(cropped_channel)

    # Reshape the cropped images
    cropped_images = torch.stack(cropped_images, dim=0)
    cropped_images = cropped_images.view(images.shape[0], images.shape[1], crop_width, crop_width)

    return cropped_images

def abs_helper(x, axis=1, is_normalization=True):
    x = torch.sqrt(torch.sum(x ** 2, dim=axis, keepdim=True))

    if is_normalization:
        for i in range(x.shape[0]):
            x[i] = (x[i] - torch.min(x[i])) / (torch.max(x[i]) - torch.min(x[i]) + 1e-16)

    x = x.to(torch.float32)

    return x