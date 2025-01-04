"""
This script trains a model using Defensive Distillation. It uses the a VGG network trained for animal classification 
as a teacher network and trains the student network. The student network is not trained on the original laberls but 
on labels obtained by applying softmax, with different temperatures, to the logits of the teacher network.
"""

from networks.vgg10 import VGG10, VGG10_lighter
from networks.inception_v3 import InceptionV3

import os, torch, random, shutil
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
from torchvision import transforms as T

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
    temperatures = [1, 5, 10, 20, 40, 50]
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


if __name__ == "__main__":

    model = VGG10_lighter(num_classes=10)
    pretrained_path = "checkpoints/VGG10lightweight_10epchs1e-4_5epochs1e-5.pth"
    device = "cuda"

    compute_distilled_labels(model, device, pretrained_path)
