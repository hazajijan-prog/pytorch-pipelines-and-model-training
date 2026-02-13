# Torch - tensor-operationer 
# nn - loss function 
# optimizer (hur modellen lär sig)

import torch 
import torch.nn as nn 
import torch.optim as optim 

# Importerar din egen modellklass från filen model.py
from src.model import SimpleClassifier

# Funktion som tränar modellen
# train_loader = data som kommer i batchar
# epochs = hur många gånger modellen ska gå igenom hela datan
def train_model(train_loader, epochs=5):
    model = SimpleClassifier() # Skapar en instans av modellen (ett neuralt nätverk)
    
    # Definierar loss-funktion
    # CrossEntropyLoss används ofta för klassificering
    # Den mäter hur fel modellens förutsägelser är
    criterion = nn.CrossEntropyLoss()
    
    # Skapar en optimizer (Adam)
    # Den uppdaterar modellens vikter för att minska felet
    optimizer = optim.Adam(model.parameters())
    
    # Loopar över antal epochs (hela datasetet flera gånger)
    for epoch in range(epochs):
        
        # Loopar över batchar av bilder och labels
        for images, labels in train_loader:
            outputs = model(images)
            loss = criterion(outputs,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print(f"Epoch {epoch+1}/{epochs} klar")


# testing 
from src.dataset import get_dataloaders

if __name__ == "__main__":
    train_loader, _ = get_dataloaders()
    train_model(train_loader, epochs=1)
