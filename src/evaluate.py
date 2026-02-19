"""
Evaluate-modul.

Innehåller funktion för att utvärdera en tränad modell på testdata.
"""

import torch


def evaluate_model(model, test_loader, device):
    """
    Utvärderar modellen på testdata och beräknar accuracy.

    Parametrar:
        model (nn.Module): Den tränade modellen.
        test_loader (DataLoader): DataLoader för testdata.
        device (torch.device): CPU eller GPU.

    Returnerar:
        accuracy (float): Modellens klassificeringsnoggrannhet.
    """

    model.eval()

    correct = 0
    total = 0

    # Stänger av gradientberäkning vid utvärdering
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            # Hämta klass med högst sannolikhet
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total if total > 0 else 0
    return accuracy
