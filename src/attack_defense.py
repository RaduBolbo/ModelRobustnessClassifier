from dataclasses import dataclass
from networks.vgg10 import VGG10_lighter
from dataloaders import get_dls
from hyperparameters import *
from pgd_attack import pgd_attack, denorm
from ddn_attack import ddn_attack
import json
from torchvision.transforms import ToPILImage
from PIL import Image, ImageFilter
import torch.nn as nn

import torch
import os
import tqdm
import numpy as np
from typing import Union

torch.manual_seed(2024)
norm = T.Normalize(mean=mean, std=std)

ATTACKS = ["PGD", "DDN"]
DEFENECES = ["Distillation", "Feat_squeezing", None]


@dataclass
class PGDParams:
    epsilon: float
    step_size: float
    num_iterations: int


@dataclass
class DDNParams:
    alpha: float
    gamma: float
    num_iterations: int


@dataclass
class ConfusionMatrix:
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0

    def update(self, attack_successful, attack_detected):
        if attack_detected and attack_successful:
            # attack correctly detected
            self.tp += 1
        elif attack_detected and not attack_successful:
            # input incorrectly flagged as attack
            self.fp += 1  #
        elif not attack_detected and not attack_successful:
            # input correctly flagged as no attack
            self.tn += 1
        elif not attack_detected and attack_successful:
            # input incorrectly flagged as no attack
            self.fn += 1

    def get_accuracy(self):
        total = self.tp + self.tn + self.fp + self.fn
        accuracy = (self.tp + self.tn) / total if total > 0 else 0
        return accuracy

    def get_precision(self):
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0
        return precision

    def get_recall(self):
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0
        return recall


class AttackDefenseClassifier:
    def __init__(
        self,
        baseline_model_path,
        dataset_root,
        attack="PGD",
        pgd_params=PGDParams(0.01, 0.001, 250),  # epsilon, step_size, num_iterations
        ddn_params=DDNParams(0.05, 0.2, 300),  # alpha, gamma, num_iterations
        defence=None,
        distillation_model=None,
        distillation_temp=1,
        device="cuda",
        out_dir="results/",
    ):
        self.device = device
        self.baseline_model = self._load_model(baseline_model_path)
        self.val_dataloader = self._load_val_dataloader(dataset_root)
        self.pgd_params = pgd_params
        self.ddn_params = ddn_params
        self.distillation_model = distillation_model
        self.distillation_temp = distillation_temp
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

        print(f"Number of validation samples: {len(self.val_dataloader)}")

        if attack in ATTACKS:
            self.attack = attack
        else:
            ValueError(f"Choose one of the following attacks: {ATTACKS}")

        if defence in DEFENECES:
            self.defence = defence
        else:
            ValueError(f"Choose one of the following defences: {DEFENECES}")

        self.total_attacks = 0
        # number of correctly classified samples after attack
        self.correct_after_attack = 0
        # number of correctly classified samples after defence (Distillation)
        self.correct_after_defence = 0
        # confusion matrix for Feature Squeezing
        self.confusion_matrix = ConfusionMatrix()
        self.attack_successful = 0

    def _load_model(self, model_path):
        """Loads model."""
        if os.path.exists(model_path):
            model = VGG10_lighter(num_classes=10)
            state_dict = torch.load(model_path)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()

            return model
        else:
            ValueError("Model path does not exist.")

    def _load_val_dataloader(self, dataset_root):
        "Loads validation dataloader."
        batch_size = 1
        num_workers = 1

        if os.path.exists(dataset_root):
            _, val_dl, classes, cls_counts = get_dls(
                root=dataset_root,
                train_transformations=train_tfs,
                val_transformations=val_tfs,
                batch_size=batch_size,
                split=[0.99, 0.01],
                num_workers=num_workers,
                feature_squeezing=True,
            )

            return val_dl
        else:
            ValueError("Dataset path does not exist.")

    def _attack(self, model, sample):
        """Perturbs the input with PGD or DNN."""

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
        # data_denorm = denorm(img)

        # perform attack
        if self.attack == "PGD":
            perturbed_img = pgd_attack(
                model,
                img,
                label,
                self.pgd_params.epsilon,
                self.pgd_params.step_size,
                self.pgd_params.num_iterations,
            )
            # perturbed_img = norm(perturbed_img)
        elif self.attack == "DDN":
            perturbed_data, success, _, last_perturbed_data = ddn_attack(
                model,
                img,
                label,
                self.ddn_params.num_iterations,
                self.ddn_params.alpha,
                self.ddn_params.gamma,
            )

            if success:
                perturbed_img = last_perturbed_data
            else:
                perturbed_img = perturbed_data

        return (True, perturbed_img)

    def _resample_img(self, img, scale_factor=0.5):
        new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
        img = img.resize(new_size, Image.BILINEAR)
        img = img.resize((img.width * 2, img.height * 2), Image.BILINEAR)

        return img

    def _bit_reduction(self, img, bits=4):
        img_np = np.array(img) / 255.0
        img_np = np.round(img_np * (2**bits - 1))
        img_np = (img_np / (2**bits - 1) * 255.0).astype(np.uint8)
        return Image.fromarray(img_np)

    def _tensor_to_pil(self, tensor):
        return ToPILImage()(denorm(tensor).detach().cpu().squeeze())

    def _get_squeezed_softmax(self, model, img, sqz_method="median_filter"):
        if sqz_method == "median_filter":
            sqz_img = img.filter(ImageFilter.MedianFilter(size=3))
        elif sqz_method == "bit_reduction":
            sqz_img = self._bit_reduction(img, 4)
        elif sqz_method == "resampling":
            sqz_img = self._resample_img(img)

        # convert to tensor
        sqz_tensor = val_tfs(sqz_img).unsqueeze(0).to(self.device)

        # inference and softmax
        sqz_softmax = (
            nn.functional.softmax(model(sqz_tensor)).detach().cpu().numpy().squeeze()
        )

        return sqz_softmax

    def _defend_feat_squeezing(self, model, perturbed_img, initial_pred, thresh=1.0):
        """Defends model with Feature squeezing. Return False if no attack is detected and true if attack is detected"""
        # median filter and resampling for feature squeezing
        perturbed_pil = self._tensor_to_pil(perturbed_img)
        median_softmax = self._get_squeezed_softmax(
            model, perturbed_pil, "median_filter"
        )
        resampled_softmax = self._get_squeezed_softmax(
            model, perturbed_pil, "resampling"
        )
        reduced_softmax = self._get_squeezed_softmax(
            model, perturbed_pil, "bit_reduction"
        )

        # compute L1-norm
        init_pred_softmax = (
            nn.functional.softmax(initial_pred).detach().cpu().numpy().squeeze()
        )
        median_l1 = np.linalg.norm(init_pred_softmax - median_softmax, ord=1)
        resampled_l1 = np.linalg.norm(init_pred_softmax - resampled_softmax, ord=1)
        reduced_l1 = np.linalg.norm(init_pred_softmax - reduced_softmax, ord=1)
        max_l1 = max(median_l1, resampled_l1, reduced_l1)

        if max_l1 < thresh:
            # no attack detected
            return False
        else:
            # attack detected
            return True

    def _defend_distillation(self, model_path, image):
        model = self._load_model(model_path)
        out = model(image)
        pred = out.max(1, keepdim=True)[1].item()

        return pred

    def _get_attack_filename(self):
        if self.attack == "PGD":
            return f"{self.attack}_eps{self.pgd_params.epsilon}_step{self.pgd_params.step_size}_iter{self.pgd_params.num_iterations}"
        elif self.attack == "DDN":
            return f"{self.attack}_alpha{self.ddn_params.alpha}_gamma{self.ddn_params.gamma}_iter{self.ddn_params.num_iterations}"

    def _compute_metrics(self):
        """Dumps metrics in .json file for Feature Squeezing"""
        results = {}
        attack_metrics = {}
        defense_metrics = {}

        results["total_samples"] = len(self.val_dataloader)
        results["total_attacks"] = self.total_attacks

        # attack metrics
        attack_name = self._get_attack_filename()
        attack_metrics["attack"] = attack_name
        results["successful_attacks"] = self.attack_successful
        attack_metrics["correct_after_attack"] = self.correct_after_attack
        attack_metrics["model_accuracy"] = (
            self.correct_after_attack / self.total_attacks
        )
        attack_metrics["comment"] = (
            "This represents the number of samples correctly classified by the model after attack."
        )

        if self.defence == "Feat_squeezing":
            defense_name = self.defence
            # compute accuracy, precision and recall for defense
            defense_metrics["defence"] = self.defence
            defense_metrics["TP_defense"] = self.confusion_matrix.tp
            defense_metrics["TN_defense"] = self.confusion_matrix.tn
            defense_metrics["FP_defense"] = self.confusion_matrix.fp
            defense_metrics["FN_defense"] = self.confusion_matrix.fn

            defense_metrics["accuracy"] = self.confusion_matrix.get_accuracy()
            defense_metrics["precision"] = self.confusion_matrix.get_precision()
            defense_metrics["recall"] = self.confusion_matrix.get_recall()
            defense_metrics["comment"] = "Attack detection metrics"

        elif self.defence == "Distillation":
            defense_name = f"{self.defence}_temp{self.distillation_temp}"
            defense_metrics["defence"] = self.defence
            defense_metrics["correct after defense"] = self.correct_after_defence
            defense_metrics["model_accuracy"] = (
                self.correct_after_defence / self.total_attacks
            )
            defense_metrics["comment"] = "Model classification accuracy after defense."

        results["attack_metrics"] = attack_metrics
        results["defence_metrics"] = defense_metrics

        filename = attack_name + "_" + defense_name + ".json"
        out_file = os.path.join(self.out_dir, filename)
        with open(out_file, "w") as json_file:
            json.dump(results, json_file, indent=4)

    def attack_defend(self):
        for sample in tqdm.tqdm(self.val_dataloader, desc="Validation set"):
            gt = sample["qry_gt"].item()
            attack_successful = False

            can_attack, perturbed_img = self._attack(self.baseline_model, sample)

            # attack samples where model is right
            if can_attack:
                self.total_attacks += 1
                attacked_output = self.baseline_model(perturbed_img)
                attacked_pred = attacked_output.max(1, keepdim=True)[1].item()

                # check if the attack was successful
                if attacked_pred == gt:
                    self.correct_after_attack += 1
                    attack_successful = False
                else:
                    self.attack_successful += 1
                    attack_successful = True
            else:
                continue

            if self.defence == "Feat_squeezing":
                attack_detected = self._defend_feat_squeezing(
                    self.baseline_model, perturbed_img, attacked_output
                )
                self.confusion_matrix.update(attack_successful, attack_detected)

            elif self.defence == "Distillation":
                pred = self._defend_distillation(self.distillation_model, perturbed_img)

                if pred == gt:
                    self.correct_after_defence += 1

        self._compute_metrics()

        print(f"Number of attacks: {self.total_attacks}")
        print(
            f"Number of correct samples after {self.attack} attack (no defence): {self.correct_after_attack}"
        )
        print(f"SUccessful attacks: {self.attack_successful}")


if __name__ == "__main__":
    model = VGG10_lighter(num_classes=10)
    pretrained_path = "checkpoints/VGG10lightweight_10epchs1e-4_5epochs1e-5.pth"
    dataset_root = "/home/ModelRobustnessClassifier/dataset/raw-img"

    # pgd_hyperparameters
    pgd_epsilon = [0.01, 0.005, 0.015, 0.0025, 0.0075, 0.0125, 0.0175]
    pgd_iter = 100
    pgd_step_size = 0.0001

    distillation_paths = [
        "/home/ModelRobustnessClassifier/src/checkpoints_distillation/student_temp20.pth",
        "/home/ModelRobustnessClassifier/src/checkpoints_distillation/student_temp50.pth",
        "/home/ModelRobustnessClassifier/src/checkpoints_distillation/student_temp70.pth",
    ]
    temps = [20, 50, 70]

    # attack with pgd
    for eps in pgd_epsilon:
        pgd_params = PGDParams(eps, pgd_step_size, pgd_iter)
        # defence: Feature Squeezing
        processor_sqz = AttackDefenseClassifier(
            pretrained_path,
            dataset_root,
            attack="PGD",
            pgd_params=pgd_params,
            defence="Feat_squeezing",
        )
        processor_sqz.attack_defend()

        # defence: distillation
        # for temp, path in zip(temps, distillation_paths):
        #     processor = AttackDefenseClassifier(
        #         pretrained_path,
        #         dataset_root,
        #         attack="PGD",
        #         defence="Distillation",
        #         pgd_params=pgd_params,
        #         distillation_model=path,
        #         distillation_temp=temp,
        #     )
        #     processor.attack_defend()

    # pgd_params = PGDParams(0.005, pgd_step_size, pgd_iter)
    # processor = AttackDefenseClassifier(
    #     pretrained_path,
    #     dataset_root,
    #     attack="DDN",
    #     defence="Distillation",
    #     pgd_params=pgd_params,
    #     distillation_model="/home/ModelRobustnessClassifier/src/checkpoints_distillation/student_temp50.pth",
    #     distillation_temp=50,
    # )
    # processor.attack_defend()
