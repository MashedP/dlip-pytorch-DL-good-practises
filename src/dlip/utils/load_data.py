import torchvision
from torchvision import transforms

def load_datasets():
    train_set = torchvision.datasets.USPS(
        root="USPS/", train=True, transform=transforms.ToTensor(), download=False
    )
    test_set = torchvision.datasets.USPS(
        root="USPS/", train=True, transform=transforms.ToTensor(), download=False
    )
    return train_set, test_set
