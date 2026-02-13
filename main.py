from src.dataset import get_dataloaders

def main():
    train_loader, test_loader = get_dataloaders()
    
    images, labels = next(iter(train_loader))
    print("Image shape:", images.shape)
    print("Train batches:", len(train_loader))
    print("Test batches:", len(test_loader))


if __name__ == "__main__":
    main()
