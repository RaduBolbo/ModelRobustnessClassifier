from hyperparameters import *
from dataloaders import get_dls
from pgd_attack import pgd_attack, denorm
from networks.vgg10 import VGG10_lighter

import torch
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from torchvision.transforms import ToPILImage
from PIL import Image, ImageFilter

torch.manual_seed(2024)


def perform_pgd(model, sample):
    # pgd attack parameters
    epsilon = 0.01
    step_size = 0.001
    num_iterations = 250

    img = sample["qry_im"].to(device)
    label = sample["qry_gt"].to(device)
    img.requires_grad = True

    # forward pass
    output = model(img)
    init_pred = output.max(1, keepdim=True)[1]  # get the prediction

    # don't attack if label is already wrong
    if init_pred.item() != label.item():
        return False, torch.zeros([1, 3, size, size])

    # denormalize
    data_denorm = denorm(img)

    # perform PGD
    perturbed_data = pgd_attack(
        model, data_denorm, label, epsilon, step_size, num_iterations
    )

    # renormalize, because the image was denormalized and now it has to be sent to the nn again
    perturbed_data_normalized = T.Normalize(mean=mean, std=std)(perturbed_data)

    return (True, perturbed_data_normalized)


def tensor_to_pil(img_tensor):
    return ToPILImage()(img_tensor.detach().cpu().squeeze())


def resample_img(img, scale_factor=0.5):
    new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
    img = img.resize(new_size, Image.BILINEAR)
    img = img.resize((img.width * 2, img.height * 2), Image.BILINEAR)

    return img


def feat_squeezing(model, model_path, dataset_root, thresh=1.0):
    batch_size = 1
    num_workers = 1

    model = model.to(device)
    if model_path is not None:
        state_dict = torch.load(
            model_path
        )  # Load the state dictionary from the .pth file
        model.load_state_dict(state_dict)
    model.eval()

    _, val_dl, classes, cls_counts = get_dls(
        root=dataset_root,
        train_transformations=train_tfs,
        val_transformations=val_tfs,
        batch_size=batch_size,
        split=[0.8, 0.2],
        num_workers=num_workers,
        feature_squeezing=True,
    )

    correct = 0
    total_attacks = 0
    successful_attacks_count = 0
    attack_detected_count = 0

    succesful_attack = False
    attack_detected = False
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for sample in tqdm(val_dl, desc="Validation"):
        can_attack, perturbed_img = perform_pgd(model, sample)

        if not can_attack:
            print("Can't attack for current sample, model is already wrong. Moving on.")
        else:
            total_attacks += 1
            output = model(perturbed_img.to(device))
            pred = output.max(1, keepdim=True)[1].item()

            if pred == sample["qry_gt"].item():
                succesful_attack = False
            else:
                succesful_attack = True
                successful_attacks_count += 1

            # squeezed inputs
            perturbed_img_pil = tensor_to_pil(denorm(perturbed_img))
            perturbed_img_median = perturbed_img_pil.filter(
                ImageFilter.MedianFilter(size=3)
            )
            perturbed_img_resample = resample_img(perturbed_img_pil)

            # apply validation transformations
            median_normalized = val_tfs(perturbed_img_median).to(device).unsqueeze(0)
            resample_normalized = (
                val_tfs(perturbed_img_resample).to(device).unsqueeze(0)
            )

            # run inference on squeezed inputs
            median_out = model(median_normalized)
            resample_out = model(resample_normalized)

            out_softmax = nn.functional.softmax(output).detach().cpu().numpy().squeeze()
            median_softmax = (
                nn.functional.softmax(median_out).detach().cpu().numpy().squeeze()
            )
            resampled_softmax = (
                nn.functional.softmax(resample_out).detach().cpu().numpy().squeeze()
            )

            median_l1 = np.linalg.norm(out_softmax - median_softmax, ord=1)
            resampled_l1 = np.linalg.norm(out_softmax - resampled_softmax, ord=1)
            l1 = max(median_l1, resampled_l1)

            if l1 < thresh:
                attack_detected = False
            else:
                attack_detected = True
                attack_detected_count += 1

            if attack_detected and succesful_attack:
                tp += 1
            elif attack_detected and not succesful_attack:
                fp += 1
            elif not attack_detected and succesful_attack:
                fn += 1
            elif not attack_detected and not succesful_attack:
                tn += 1

    print(f"Total attacks: {total_attacks}")
    print(f"Successful attacks: {successful_attacks_count}")
    print(f"Attacks detected: {attack_detected_count}")
    print(f"tp: {tp}")
    print(f"tn: {tn}")
    print(f"fp: {fp}")
    print(f"fn: {fn}")


if __name__ == "__main__":
    model = VGG10_lighter(num_classes=10)
    pretrained_path = "checkpoints/VGG10lightweight_10epchs1e-4_5epochs1e-5.pth"
    dataset_root = "/home/ModelRobustnessClassifier/dataset/raw-img"
    feat_squeezing(model, pretrained_path, dataset_root)
    print("Feature squeezing")
