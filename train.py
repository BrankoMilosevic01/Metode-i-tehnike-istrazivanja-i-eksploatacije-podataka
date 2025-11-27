import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, f1_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from utils import load_config, get_dataloaders
from model import MalariaDINOClassifier

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels, all_probs = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            probs = torch.sigmoid(outputs)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
   
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    preds = (all_probs > 0.5).astype(float)
    try: auc = roc_auc_score(all_labels, all_probs)
    except: auc = 0.5
   
    metrics = {"loss": running_loss / len(loader), "accuracy": np.mean(preds == all_labels),
               "auc_roc": auc, "f1": f1_score(all_labels, preds, zero_division=0)}
    return metrics, all_labels, all_probs

def plot_curves(labels, probs):
    fpr, tpr, _ = roc_curve(labels, probs)
    precision, recall, _ = precision_recall_curve(labels, probs)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title('ROC Curve (DINOv2)')
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='blue', lw=2, label='PR curve')
    plt.title('Precision-Recall Curve')
    plt.savefig('evaluation_curves.png')
    plt.close()

if __name__ == "__main__":
    config = load_config()
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train_loader, val_loader = get_dataloaders(config)
    model = MalariaDINOClassifier(model_name=config['model']['name']).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=config['training']['learning_rate'])
   
    val_labels, val_probs = [], []
    for epoch in range(config['training']['epochs']):
        print(f"\nEpoch {epoch+1}/{config['training']['epochs']}")
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics, val_labels, val_probs = evaluate(model, val_loader, criterion, device)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | AUC: {val_metrics['auc_roc']:.4f} | F1: {val_metrics['f1']:.4f}")
   
    print("\nGenerating curves...")
    plot_curves(val_labels, val_probs)
    print("Done! Curves saved as 'evaluation_curves.png'")
