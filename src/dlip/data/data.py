import torch
import torchvision
from dlip.data.usps import download_usps
from torch.utils.data import random_split
from torchvision import transforms


def load_dataset(path, seed=0):
    generator = torch.Generator()
    generator.manual_seed(seed)
    download_usps(path)
    dataset = torchvision.datasets.USPS(
        root=path + "USPS/", train=True, transform=transforms.ToTensor(), download=False
    )
    train_set, val_set = random_split(dataset, [6000, 1291], generator=generator)
    return train_set, val_set
