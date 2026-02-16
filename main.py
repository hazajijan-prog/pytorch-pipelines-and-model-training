import torch
from src.dataset import get_dataloaders
from src.model import SimpleClassifier
from src.train import train_model
from src.evaluate import evaluate_model

def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_dataloaders(batch_size=64)
    
    # Skapa modellen
    model = SimpleClassifier()
    model.to(device)
    
    # Träna modellen
    model = train_model(model, train_loader, device, epochs=5, lr=0.001)

    # Utvärdera modellen
    accuracy = evaluate_model(model, test_loader, device)

    print(f"Test Accuracy: {accuracy:.4f}")
    
    

if __name__ == "__main__":
    main()
