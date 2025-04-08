import os
import json
import random
import numpy as np
from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import wandb

from fractal_dimension import compute_fractal_dimension
from model import FractalResNet


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


DATASET_DIR = r'D:\fractal\datasets'
MALIGNANT_FOLDER = os.path.join(DATASET_DIR, 'malignant_tissues')
BENIGN_FOLDER   = os.path.join(DATASET_DIR, 'benign_tissues')


class TissueDataset(Dataset):
    def __init__(self, malignant_paths, benign_paths, transform=None):
        self.transform = transform
        self.data = []
        # Labeling malignant as 1, benign as 0
        for path in malignant_paths:
            self.data.append((path, 1))
        for path in benign_paths:
            self.data.append((path, 0))
        random.shuffle(self.data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image_transformed = self.transform(image)
        else:
            image_transformed = transforms.ToTensor()(image)
        
        
        image_np = np.array(image.resize((256, 256)))
        fd = compute_fractal_dimension(image_np)
        fd_tensor = torch.tensor([fd], dtype=torch.float32)
        
        label = torch.tensor(label, dtype=torch.long)
        return image_transformed, fd_tensor, label

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, fds, labels in loader:
            images = images.to(device)
            fds = fds.to(device)
            labels = labels.to(device)
            outputs = model(images, fds)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def main():
   
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    
    malignant_paths = glob(os.path.join(MALIGNANT_FOLDER, '*'))
    benign_paths = glob(os.path.join(BENIGN_FOLDER, '*'))

    
    dataset = TissueDataset(malignant_paths, benign_paths, transform=transform)
    total = len(dataset)
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    
    wandb.init(project="cancer-tissue-fractality-detector", config={
        "epochs": 20,
        "batch_size": 32,
        "learning_rate": 1e-4,
    })
    config = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FractalResNet(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    wandb.watch(model, log="all")

    best_val_acc = 0.0

   
    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
        for images, fds, labels in pbar:
            images = images.to(device)
            fds = fds.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images, fds)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            pbar.set_postfix(loss=loss.item())
        
        epoch_loss = running_loss / len(train_loader.dataset)
        val_acc = evaluate(model, val_loader, device)
        wandb.log({
            "epoch": epoch+1,
            "train_loss": epoch_loss,
            "val_accuracy": val_acc
        })
        print(f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, Val Accuracy={val_acc:.4f}")
        
       
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
    
 
    test_accuracy = evaluate(model, test_loader, device)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    
    metrics = {
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "best_val_accuracy": best_val_acc,
        "test_accuracy": test_accuracy
    }
    with open("accuracy_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    
    wandb.log({"test_accuracy": test_accuracy})
    wandb.finish()

if __name__ == "__main__":
    main()
