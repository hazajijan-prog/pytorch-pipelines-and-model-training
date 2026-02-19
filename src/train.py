"""
Train-modul.

Innehåller funktion för att träna en modell i PyTorch.
"""

import torch.nn as nn
import torch.optim as optim


def train_model(model, train_loader, device, epochs=1, lr=0.001):
    """
    Tränar en modell på angiven träningsdata.

    Parametrar:
        model (nn.Module): Modellen som ska tränas.
        train_loader (DataLoader): DataLoader för träningsdata.
        device (torch.device): CPU eller GPU.
        epochs (int): Antal träningsomgångar.
        lr (float): Learning rate för optimeraren.

    Returnerar:
        model (nn.Module): Den tränade modellen.
    """

    model.train()

    # Definiera loss-funktion och optimerare
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        epoch_loss = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Räkna genomsnittlig loss för epoken
        avg_loss = epoch_loss / len(train_loader)

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

    return model
