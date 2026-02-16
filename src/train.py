# Torch - tensor-operationer 
# nn - loss function 
# optimizer (hur modellen lär sig)

import torch 
import torch.nn as nn 
import torch.optim as optim 

# Funktion som tränar modellen
# train_loader = data som kommer i batchar
# epochs = hur många gånger modellen ska gå igenom hela datan
def train_model(train_loader, epochs=1, lr=0.001):
    
    # Sätt modellen i train-läge
    model.train()
    
    # Definierar loss-funktion
    # CrossEntropyLoss används ofta för klassificering
    # Den mäter hur fel modellens förutsägelser är
    criterion = nn.CrossEntropyLoss()
    
    # Skapar en optimizer (Adam)
    # Den uppdaterar modellens vikter för att minska felet
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Loopar över antal epochs (hela datasetet flera gånger)
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
            
            # Lägg till batchens loss
            epoch_loss += loss.item()
        
        # Räkna genomsnittlig loss för epoken
        avg_loss = epoch_loss / len(train_loader)
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        
    return model 
