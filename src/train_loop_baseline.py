"""
This is a trainign script which deosn't employ defence
"""

from networks.vgg10 import VGG10, VGG10_lighter, VGG14_lighterOnlyOneDense
from networks.inception_v3 import InceptionV3
from dataloaders import get_dls
from hyperparameters import *

import os, random, shutil
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(2024)
Image.MAX_IMAGE_PIXELS = None


def compute_class_weights(cls_counts):
    total_samples = sum(cls_counts.values())
    class_weights = {cls: total_samples / count for cls, count in cls_counts.items()}
    weights_tensor = torch.tensor(
        [class_weights[cls] for cls in sorted(cls_counts.keys())], dtype=torch.float
    )
    return weights_tensor


def train_baseline(model, device, pretraiend_path=None):

    writer = SummaryWriter(log_dir="src/runs")
    tr_dl, val_dl, classes, cls_counts = get_dls(
        root=dataset_root,
        train_transformations=train_tfs,
        val_transformations=val_tfs,
        batch_size=batch_size,
        split=[0.8, 0.2],
        num_workers=num_workers,
    )

    print(len(tr_dl))
    print(len(val_dl))
    print(classes)
    print(cls_counts)

    model = model.to(device)
    if pretraiend_path is not None:
        state_dict = torch.load(
            pretraiend_path
        )  # Load the state dictionary from the .pth file
        model.load_state_dict(state_dict)

    # Compute class weights
    class_weights = compute_class_weights(cls_counts).to(device)

    # Loss function with weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Track metrics
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        if epoch == 10:
            new_lr = 1e-5
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            print(f"Learning rate updated to {new_lr}")

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
        train_loss = train_loss / len(tr_dl)
        train_f1_scores = f1_score(all_labels_train, all_preds_train, average=None)

        print(
            f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}"
        )
        for cls_idx, f1 in enumerate(train_f1_scores):
            print(f"F1 Score for Class {cls_idx}: {f1:.4f}")

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_accuracy, epoch)

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
        val_loss = val_loss / len(val_dl)
        val_f1_scores = f1_score(all_labels_val, all_preds_val, average=None)
        print(
            f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}"
        )
        for cls_idx, f1 in enumerate(val_f1_scores):
            print(f"F1 Score for Class {cls_idx}: {f1:.4f}")

        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_accuracy, epoch)

        torch.save(model.state_dict(), f"checkpoints/model_{epoch}.pth")

    print("Training complete.")
    writer.flush()
    writer.close()


if __name__ == "__main__":

    # model = VGG10(num_classes=10)
    model = VGG10_lighter(num_classes=10)
    # model = VGG14_lighterOnlyOneDense(num_classes=10)

    pretrained_path = "checkpoints/VGG10lightweight_10epchs1e-4_5epochs1e-5.pth"
    # pretrained_path = 'checkpoints\model_2.pth'

    train_baseline(model, device)
