"""
This script trains a model using Defensive Distillation. It uses the a VGG network trained for animal classification 
as a teacher network and trains the student network. The student network is not trained on the original laberls but 
on labels obtained by applying softmax, with different temperatures, to the logits of the teacher network.
"""

from networks.vgg10 import VGG10, VGG10_lighter
from networks.inception_v3 import InceptionV3
from hyperparameters import *
from train_loop_baseline import compute_class_weights

import os, torch, random, shutil
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
from torchvision import transforms as T
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

torch.manual_seed(2024)
from dataloaders import get_dls

import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from sklearn.metrics import f1_score
import csv


def loss_function(logits, target):
    """CrossEntropyLoss. The pytorch implementation applies
    softmax internally but for distillation defense,
    softmax with temperature is needed."""
    if target.ndimension() == 1:
        # needs one hot encoding (for validation; labels are not soft but categorical)
        one_hot = torch.zeros(target.size(0), 10, dtype=torch.float32).to(device)
        one_hot.scatter_(1, target.unsqueeze(1), 1)
        return -(one_hot * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
    else:
        return -(target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def train(
    train_dl,
    val_dl,
    class_weights,
    device="cuda",
    temperature=50,
    model_type="teacher",
    teacher_model=None,
):
    if model_type == "student" and teacher_model is None:
        ValueError("Provide a teacher model.")

    print(f"Training distillation {model_type} model for temperature {temperature}")

    # define paths
    runs_path = os.path.join("src/runs", f"distil_temp_{temperature}")
    model_dir = os.path.join(f"src/checkpoints_distillation/")
    os.makedirs(model_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=runs_path)
    # instantiate model
    model = VGG10_lighter(num_classes=10)
    model = model.to(device)

    if model_type == "student":
        # load teacher model
        teacher_model = teacher_model.to(device)
        teacher_model.eval()

    criterion = loss_function
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Track metrics
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # decrtease learning rate fater 10 epochs
        if epoch == 10:
            new_lr = 1e-5
            for param_group in optimizer.param_groups:
                param_group["lr"] = new_lr
            print(f"Learning rate updated to {new_lr}")

        # Training loop
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        all_preds_train = []
        all_labels_train = []

        for batch in tqdm(train_dl, desc="Training"):
            optimizer.zero_grad()
            qry_im = batch["qry_im"].to(device)

            if model_type == "teacher":
                qry_gt = batch["qry_gt"].to(device)
            else:
                # model is student, get soft labels
                out = teacher_model(qry_im)
                qry_gt = nn.functional.softmax(out / temperature, dim=1)

            outputs = model(qry_im)
            loss = criterion(outputs / temperature, qry_gt)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)

            if model_type == "student":
                qry_gt = torch.max(qry_gt, 1)[1]

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

    model_path = os.path.join(model_dir, f"{model_type}_temp{temperature}.pth")
    torch.save(model.state_dict(), model_path)

    print("Training complete.")
    writer.flush()
    writer.close()

    return model_path


if __name__ == "__main__":

    device = "cuda"
    tr_dl, val_dl, classes, cls_counts = get_dls(
        root=dataset_root,
        train_transformations=train_tfs,
        val_transformations=val_tfs,
        batch_size=batch_size,
        split=[0.8, 0.2],
        num_workers=num_workers,
    )
    class_weights = compute_class_weights(cls_counts).to(device)
    temps = [20, 50, 70]

    for temp in temps:
        # train teacher
        teacher_path = train(
            tr_dl,
            val_dl,
            class_weights,
            device="cuda",
            temperature=temp,
            model_type="teacher",
            teacher_model=None,
        )

        # train student
        teacher_network = VGG10_lighter(num_classes=10)
        state_dict = torch.load(teacher_path)
        teacher_network.load_state_dict(state_dict)

        student_path = train(
            tr_dl,
            val_dl,
            class_weights,
            device="cuda",
            temperature=temp,
            model_type="student",
            teacher_model=teacher_network,
        )
