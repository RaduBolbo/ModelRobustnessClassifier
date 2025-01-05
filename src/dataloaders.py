import os, torch, random, shutil, numpy as np, pandas as pd
from glob import glob
from PIL import Image, ImageFilter

Image.MAX_IMAGE_PIXELS = None
from torch.utils.data import random_split, Dataset, DataLoader
from torchvision import transforms as T

torch.manual_seed(2024)
import cv2
import csv
import tqdm
from collections import Counter
from hyperparameters import *


class CustomDataset(Dataset):

    def __init__(
        self,
        root,
        transformations=None,
        distilled_labels_path=None,
        feature_squeezing_flag=False,
    ):

        self.transformations = transformations
        self.im_paths = glob(f"{root}/*/*.jpeg")
        self.cls_names, self.cls_counts, count = {}, {}, 0
        # self.distilled_labels = {}
        self.distillation_defense_flag = False
        self.feature_squeezing_flag = feature_squeezing_flag

        for idx, im_path in enumerate(self.im_paths):
            cls_name = self.get_cls_name(im_path)
            if cls_name not in self.cls_names:
                self.cls_names[cls_name] = count
                count += 1
            if cls_name not in self.cls_counts:
                self.cls_counts[cls_name] = 1
            else:
                self.cls_counts[cls_name] += 1

        if distilled_labels_path is not None:
            self.distilled_labels = self.get_distilled_labels(distilled_labels_path)
            self.distillation_defense_flag = True

    def get_cls_name(self, path):
        return os.path.dirname(path).split("/")[-1]

    def __len__(self):
        return len(self.im_paths)

    def get_pos_neg_im_paths(self, qry_label):

        pos_im_paths = [
            im_path
            for im_path in self.im_paths
            if qry_label == self.get_cls_name(im_path)
        ]
        neg_im_paths = [
            im_path
            for im_path in self.im_paths
            if qry_label != self.get_cls_name(im_path)
        ]

        pos_rand_int = random.randint(a=0, b=len(pos_im_paths) - 1)
        neg_rand_int = random.randint(a=0, b=len(neg_im_paths) - 1)

        return pos_im_paths[pos_rand_int], neg_im_paths[neg_rand_int]

    def get_distilled_labels(self, distilled_path):
        distilled_labels = {}

        with open(distilled_path, mode="r") as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                # Extract the image path and values
                img_path = row[0]
                values = list(map(float, row[1:]))
                distilled_labels[img_path] = np.array(values, dtype=np.float32)

        return distilled_labels

    def resample_img(self, img, scale_factor=0.5):
        new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
        img = img.resize(new_size, Image.BILINEAR)
        img = img.resize((img.width * 2, img.height * 2), Image.BILINEAR)

        return img

    def __len__(self):
        return len(self.im_paths)

    def __getitem__(self, idx):

        im_path = self.im_paths[idx]
        original_image = Image.open(im_path).convert("RGB")
        qry_label = self.get_cls_name(im_path)

        pos_im_path, neg_im_path = self.get_pos_neg_im_paths(qry_label=qry_label)
        pos_im, neg_im = Image.open(pos_im_path).convert("RGB"), Image.open(
            neg_im_path
        ).convert("RGB")

        qry_gt = self.cls_names[qry_label]
        neg_gt = self.cls_names[self.get_cls_name(neg_im_path)]

        if self.transformations is not None:
            qry_im = self.transformations(original_image)
            pos_im = self.transformations(pos_im)
            neg_im = self.transformations(neg_im)

        data = {}

        data["qry_im"] = qry_im
        data["qry_gt"] = qry_gt
        data["pos_im"] = (
            pos_im  # **** not used for the moment. For normal trining only query is used
        )
        data["neg_im"] = (
            neg_im  # **** not used for the moment. For normal trining only query is used
        )
        data["neg_gt"] = (
            neg_gt  # **** not used for the moment. For normal trining only query is used
        )

        if self.distillation_defense_flag:
            distilled_label = self.distilled_labels[im_path]
            data["distilled_label"] = distilled_label

        if self.feature_squeezing_flag:
            median_img = original_image.filter(ImageFilter.MedianFilter(size=3))
            resampled_img = self.resample_img(original_image)
            data["median_filter_img"] = self.transformations(median_img)
            data["resampled_img"] = self.transformations(resampled_img)

        return data


def get_class_count(dataset, subset):
    all_img_paths = np.array(dataset.im_paths)
    subset_indices = subset.indices
    subset_paths = all_img_paths[subset_indices].tolist()
    subset_labels = list(map(lambda x: os.path.dirname(x).split("/")[-1], subset_paths))
    subset_counts = dict(Counter(subset_labels))

    return subset_counts


def compute_percentage_split(dataset, train_ds, val_ds):
    train_counts = get_class_count(dataset, train_ds)
    val_counts = get_class_count(dataset, val_ds)

    for key in train_counts.keys():
        total_samples = train_counts[key] + val_counts[key]
        train_percenatge = "{0:.2f}".format(train_counts[key] / total_samples * 100)
        val_percentage = "{0:.2f}".format(val_counts[key] / total_samples * 100)
        print(
            f"For class {key} there are {train_counts[key]} ({train_percenatge}%) training samples and {val_counts[key]} ({val_percentage}%) validation samples."
        )


def get_dls(
    root,
    train_transformations,
    val_transformations,
    batch_size,
    split=[0.8, 0.2],
    num_workers=4,
    distilled_labels_path=None,
    feature_squeezing=False,
):

    ds = CustomDataset(
        root=root,
        distilled_labels_path=distilled_labels_path,
        feature_squeezing_flag=feature_squeezing,
    )

    total_len = len(ds)
    tr_len = int(total_len * split[0])
    vl_len = total_len - tr_len

    tr_ds, vl_ds = random_split(dataset=ds, lengths=[tr_len, vl_len])
    compute_percentage_split(ds, tr_ds, vl_ds)

    # add the propper transformations
    tr_ds.dataset.transformations = train_transformations
    vl_ds.dataset.transformations = val_transformations

    tr_dl, val_dl = DataLoader(
        tr_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    ), DataLoader(vl_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return tr_dl, val_dl, ds.cls_names, ds.cls_counts


def display_tensor(tensor):
    print(tensor.shape)
    array = tensor.cpu().numpy()
    print(np.min(array), np.max(array))
    array = (array - np.min(array)) / (np.max(array) - np.min(array))  # normalize
    array = np.transpose(array, (1, 2, 0))
    print(np.min(array), np.max(array))

    cv2.imshow("Image", array)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    root = "dataset/raw-img"
    mean, std, size, batch_size = (
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
        224,
        16,
    )  # media, deviatai standard, diemsniuenaimaginilor si batch size-ul
    tfs = T.Compose(
        [
            T.ToTensor(),
            T.Resize(size=(size, size), antialias=False),
            T.Normalize(mean=mean, std=std),
        ]
    )
    tr_dl, val_dl, classes, cls_counts = get_dls(
        root, train_tfs, val_tfs, batch_size=batch_size
    )

    print(len(tr_dl))
    print(len(val_dl))
    print(classes)
    print(cls_counts)

    for batch in tr_dl:
        print(batch["qry_im"].shape)
        print(batch["qry_gt"].shape)
        print(batch["pos_im"].shape)
        print(batch["neg_im"].shape)
        print(batch["neg_gt"].shape)
        print(batch["qry_gt"])
        print(batch["neg_gt"])

        display_tensor(batch["qry_im"][0, :, :, :])
        display_tensor(batch["pos_im"][0, :, :, :])
        display_tensor(batch["neg_im"][0, :, :, :])
        break
