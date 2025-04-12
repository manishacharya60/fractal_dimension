import os
import json
import random
import numpy as np
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import wandb
from sklearn.metrics import f1_score

from fractal_dimension import compute_fractal_dimension
from model import FractalResNet


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


DATASET_DIR = r'output' # raja yesma la: update dataset directory to the provided output folder
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
TEST_DIR = os.path.join(DATASET_DIR, "test")

class TissueDataset(Dataset):
    def __init__(self, malignant_paths, benign_paths, transform=None):
        self.transform = transform
        self.data = []
        # malignant lai 1, benign lai 0
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

def evaluate_loss_acc(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, fds, labels in loader:
            images = images.to(device)
            fds = fds.to(device)
            labels = labels.to(device)
            outputs = model(images, fds)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

def compute_test_f1(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, fds, labels in loader:
            images = images.to(device)
            fds = fds.to(device)
            outputs = model(images, fds)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    score = f1_score(all_labels, all_preds, average='weighted')
    return score

def main():
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    
    train_malignant = glob(os.path.join(TRAIN_DIR, "malignant", "*"))
    train_benign   = glob(os.path.join(TRAIN_DIR, "benign", "*"))
    val_malignant   = glob(os.path.join(VAL_DIR, "malignant", "*"))
    val_benign     = glob(os.path.join(VAL_DIR, "benign", "*"))
    test_malignant  = glob(os.path.join(TEST_DIR, "malignant", "*"))
    test_benign    = glob(os.path.join(TEST_DIR, "benign", "*"))

    train_dataset = TissueDataset(train_malignant, train_benign, transform=transform)
    val_dataset   = TissueDataset(val_malignant, val_benign, transform=transform)
    test_dataset  = TissueDataset(test_malignant, test_benign, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    
    wandb.init(project="cancer-tissue-fractality-detector", config={
        "epochs": 50,
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
    metrics_history = []  
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        total_samples = 0
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
            _, predicted = torch.max(outputs, 1)
            running_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            
            pbar.set_postfix(loss=loss.item())

       
        train_loss = running_loss / total_samples
        train_accuracy = running_correct / total_samples
        train_loss_history.append(train_loss)
        train_acc_history.append(train_accuracy)

        
        val_loss, val_accuracy = evaluate_loss_acc(model, val_loader, device, criterion)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_accuracy)

        
        wandb.log({
            "epoch": epoch+1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        })
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_accuracy:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_accuracy:.4f}")

       
        metrics_history.append({
            "epoch": epoch+1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        })
        
       
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), "best_model.pth")
    
    
    test_loss, test_accuracy = evaluate_loss_acc(model, test_loader, device, criterion)
    test_f1 = compute_test_f1(model, test_loader, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test F1 Score: {test_f1:.4f}")

    
    wandb.log({
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "test_f1_score": test_f1
    })
    wandb.finish()

    
    metrics_data = {
        "epoch_metrics": metrics_history,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "test_f1_score": test_f1,
        "train_dataset_size": len(train_dataset),
        "val_dataset_size": len(val_dataset),
        "test_dataset_size": len(test_dataset)
    }
    with open("training_metrics.json", "w") as f:
        json.dump(metrics_data, f, indent=4)
    
    
    epochs = range(1, config.epochs + 1)
    
    
    plt.figure()
    plt.plot(epochs, train_loss_history, 'b-', label="Train Loss")
    plt.plot(epochs, val_loss_history, 'r-', label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs. Epoch")
    plt.legend()
    plt.savefig("loss_vs_epoch.png")
    plt.close()
    
   
    plt.figure()
    plt.plot(epochs, train_acc_history, 'b-', label="Train Accuracy")
    plt.plot(epochs, val_acc_history, 'r-', label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Epoch")
    plt.legend()
    plt.savefig("accuracy_vs_epoch.png")
    plt.close()

if __name__ == "__main__":
    main()
