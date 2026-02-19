"""
Dataset-Modul.

Laddar ner CIFAR-10 och skapa DataLoaders för träning och testing.
"""

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os


def get_dataloaders(batch_size=64):
    """
    Skapar tränings och test för CIFAR-10.

    Parametrar:
        Batch_size (int): antal bilder per batch.

    Returnerar:
        train_loader (DataLoader): DataLoader för träningsdata.
        test_loader (DataLoader): DataLoader för testdata.
    """

    # Projektets rotkatalog och sökväg till datamappen
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = f"{ROOT_DIR}/../data"

    # Ladda träningsdatasetet
    train_dataset = datasets.CIFAR10(
        root=DATA_DIR, train=True, transform=transforms.ToTensor(), download=True
    )

    # Ladda testdatasetet
    test_dataset = datasets.CIFAR10(
        root=DATA_DIR, train=False, transform=transforms.ToTensor(), download=True
    )

    # Skapa DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
