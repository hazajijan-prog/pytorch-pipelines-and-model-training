from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision
import os

def get_dataloaders(batch_size=64):
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = f"{ROOT_DIR}/../data"

    train_dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        transform=transforms.ToTensor(),
        download=True
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        transform=transforms.ToTensor(),
        download=True
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    return train_loader, test_loader
