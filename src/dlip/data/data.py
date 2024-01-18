import torchvision
from torchvision import transforms

def load_dataset(path):
    train_set = torchvision.datasets.USPS(
        root=path+"USPS/", train=True, transform=transforms.ToTensor(), download=False
    )
    return train_set