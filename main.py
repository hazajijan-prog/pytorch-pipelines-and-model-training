"""
Main-modul.

Startpunkt för projektet. Kör hela tränings- och
utvärderingspipen med valda hyperparametrar.
"""

import torch
from src.dataset import get_dataloaders
from src.model import SimpleClassifier
from src.train import train_model
from src.evaluate import evaluate_model


def main():
    # Välj beräkningsenhet (GPU om tillgänglig, annars CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Ladda data
    train_loader, test_loader = get_dataloaders(batch_size=64)

    # Skapa modellen
    model = SimpleClassifier()
    model.to(device)

    # Träna modellen med valda hyperparametrar
    model = train_model(
        model, 
        train_loader, 
        device, 
        epochs=5, 
        lr=0.001)

    # Utvärdera modellen på testdata
    accuracy = evaluate_model(model, test_loader, device)

    print(f"Test Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
