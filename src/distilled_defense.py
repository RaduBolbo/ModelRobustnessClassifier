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
torch.manual_seed(2024)
from dataloaders import get_dls

import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from sklearn.metrics import f1_score
import csv


def compute_distilled_labels(model, device, pretraiend_path):
    if pretraiend_path is None:
        raise ValueError("pretrained_path can't be None")

    # hyperparameters
    temperatures = [1, 5, 10, 20]
    batch_size = 1
    num_workers = 1
    root = "dataset/raw-img"
    csv_fields = ["img_path", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    # create directory for saving the distilled labels
    save_dir = os.path.join(
        os.path.split(os.path.dirname(__file__))[0], "distilled_labels"
    )
    os.makedirs(save_dir, exist_ok=True)

    mean, std, size = (
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
        224,
    )  # media, deviatai standard, diemsniuenaimaginilor

    val_tfs = T.Compose(
        [
            T.ToTensor(),
            T.Resize(size=(size, size), antialias=False),
            T.Normalize(mean=mean, std=std),
        ]
    )

    tr_dl, val_dl, classes, cls_counts = get_dls(
        root=root,
        train_transformations=val_tfs,
        val_transformations=val_tfs,
        batch_size=batch_size,
        split=[1],
        num_workers=num_workers,
    )

    print(len(tr_dl))
    print(len(val_dl))
    print(classes)
    print(cls_counts)

    # load the model
    model = model.to(device)
    state_dict = torch.load(pretraiend_path)
    model.load_state_dict(state_dict)
    model.eval()

    for temp in temperatures:
        print(f"Computing labels for softmax with temperature {temp}.")
        file_name = f"labels_temp_{temp}.csv"
        file_path = os.path.join(save_dir, file_name)

        with open(file_path, "w") as f:
            writer = csv.writer(f, delimiter=",", lineterminator="\n")
            # add column names
            writer.writerow(csv_fields)

            with torch.no_grad():
                for batch in tqdm(tr_dl, desc="Validation"):
                    qry_im = batch["qry_im"].to(device)
                    img_path = batch["im_path"]

                    outputs = model(qry_im)
                    softmax_outputs = (
                        nn.functional.softmax(outputs / temp, dim=1)
                        .detach()
                        .cpu()[0]
                        .tolist()
                    )

                    softmax_outputs.insert(0, img_path[0])
                    writer.writerow(softmax_outputs)

def train_distillation_defence(model, device, distillation_path=None):
    
    _, filename = os.path.split(distillation_path)
    temp = filename.split(".")[0].split("_")[-1]
    
    print(f"Training distillation model for temperature {temp}")
    runs_path = os.path.join("src/runs", f"distil_temp_{temp}")
    model_dir = os.path.join(f"src/checkpoints/models_temp_{temp}")
    os.makedirs(model_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=runs_path)
    tr_dl, val_dl, classes, cls_counts = get_dls(
        root=dataset_root,
        train_transformations=train_tfs,
        val_transformations=val_tfs,
        batch_size=batch_size,
        split=[0.8, 0.2],
        num_workers=num_workers,
        distilled_labels_path=distillation_path
    )

    model = model.to(device)

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
            distil_label = batch["distilled_label"].to(device)

            outputs = model(qry_im)
            loss = criterion(outputs, distil_label)
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

        model_path = os.path.join(model_dir, f"model_{epoch}.pth")

        torch.save(model.state_dict(), model_path)

    print("Training complete.")
    writer.flush()
    writer.close()


if __name__ == "__main__":

    model = VGG10_lighter(num_classes=10)
    pretrained_path = "/home/ModelRobustnessClassifier/checkpoints/model_14.pth"
    device = "cuda"

    # compute_distilled_labels(model, device, pretrained_path)
    train_distillation_defence(model, device, "/home/ModelRobustnessClassifier/distilled_labels/labels_temp_10.csv")
