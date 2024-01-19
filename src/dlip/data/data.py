import torchvision
from torch.utils.data import random_split
from torchvision import transforms


def load_dataset(path):
    dataset = torchvision.datasets.USPS(
        root=path + "USPS/", train=True, transform=transforms.ToTensor(), download=False
    )
    train_set, val_set = random_split(dataset, [6000, 1291])
    return train_set, val_set
