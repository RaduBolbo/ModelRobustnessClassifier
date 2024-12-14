'''
This is a trainign script which deosn't employ defence
'''
from networks.vgg10 import VGG10, VGG10_lighter, VGG14_lighterOnlyOneDense
from networks.inception_v3 import InceptionV3

import os, torch, random, shutil, numpy as np, pandas as pd
from glob import glob; from PIL import Image
Image.MAX_IMAGE_PIXELS = None 
from torchvision import transforms as T
torch.manual_seed(2024)
from dataloaders import get_dls

import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from sklearn.metrics import f1_score


def compute_class_weights(cls_counts):
    total_samples = sum(cls_counts.values())
    class_weights = {cls: total_samples / count for cls, count in cls_counts.items()}
    weights_tensor = torch.tensor([class_weights[cls] for cls in sorted(cls_counts.keys())], dtype=torch.float)
    return weights_tensor

def train_baseline(model, device, pretraiend_path=None):
    epochs = 20
    #lr = 1e-4 # this is good
    lr = 1e-5
    batch_size = 32
    num_workers = 8

    root = "dataset/raw-img"
    mean, std, size = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 224 # media, deviatai standard, diemsniuenaimaginilor
    val_tfs = T.Compose([T.ToTensor(),
        T.Resize(size = (size, size),
        antialias = False),
        T.Normalize(mean = mean, std = std)])
    train_tfs = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=45),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
        T.ToTensor(),
        T.RandomResizedCrop(size=(224, 224), scale=(0.7, 1.2)),  
        T.Resize(size = (size, size),
        antialias = False),
        T.Normalize(mean=mean, std=std),
        ])
    tr_dl, val_dl, classes, cls_counts = get_dls(root = root, train_transformations = train_tfs, val_transformations = val_tfs, batch_size = batch_size, split = [0.8, 0.2], num_workers = num_workers)

    print(len(tr_dl))
    print(len(val_dl))
    print(classes)
    print(cls_counts)

    model = model.to(device)
    if pretraiend_path is not None:
        state_dict = torch.load(pretraiend_path)  # Load the state dictionary from the .pth file
        model.load_state_dict(state_dict)
    
    # Compute class weights
    class_weights = compute_class_weights(cls_counts).to(device)
    
    # Loss function with weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Track metrics
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        
        # Training loop
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        all_preds_train = []
        all_labels_train = []
        for batch in tqdm(tr_dl, desc="Training"):
            optimizer.zero_grad()
            
            qry_im = batch["qry_im"].to(device)
            qry_gt = batch["qry_gt"].to(device)
            
            outputs = model(qry_im)
            loss = criterion(outputs, qry_gt)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == qry_gt).sum().item()
            total_train += qry_gt.size(0)
            
            # Collect predictions and labels for F1 score
            all_preds_train.extend(preds.cpu().numpy())
            all_labels_train.extend(qry_gt.cpu().numpy())

        train_accuracy = correct_train / total_train
        train_f1_scores = f1_score(all_labels_train, all_preds_train, average=None)
        print(f"Training Loss: {train_loss / len(tr_dl):.4f}, Training Accuracy: {train_accuracy:.4f}")
        for cls_idx, f1 in enumerate(train_f1_scores):
            print(f"F1 Score for Class {cls_idx}: {f1:.4f}")

        # Validation loop
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        all_preds_val = []
        all_labels_val = []

        with torch.no_grad():
            for batch in tqdm(val_dl, desc="Validation"):
                qry_im = batch["qry_im"].to(device)
                qry_gt = batch["qry_gt"].to(device)
                
                outputs = model(qry_im)
                loss = criterion(outputs, qry_gt)
                
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == qry_gt).sum().item()
                total_val += qry_gt.size(0)
                
                # Collect predictions and labels for F1 score
                all_preds_val.extend(preds.cpu().numpy())
                all_labels_val.extend(qry_gt.cpu().numpy())

        val_accuracy = correct_val / total_val
        val_f1_scores = f1_score(all_labels_val, all_preds_val, average=None)
        print(f"Validation Loss: {val_loss / len(val_dl):.4f}, Validation Accuracy: {val_accuracy:.4f}")
        for cls_idx, f1 in enumerate(val_f1_scores):
            print(f"F1 Score for Class {cls_idx}: {f1:.4f}")


        torch.save(model.state_dict(), f"checkpoints/model_{epoch}.pth")
    print("Training complete.")
    


if __name__ == '__main__':

    #model = VGG10(num_classes=10)
    model = VGG10_lighter(num_classes=10)
    #model = VGG14_lighterOnlyOneDense(num_classes=10)
    device = 'cuda'
    #pretrained_path = 'checkpoints\VGG10lightweight_10epchs1e-4_5epochs1e-5.pth'
    pretrained_path = 'checkpoints\model_2.pth'
    
    train_baseline(model, device, pretrained_path)

