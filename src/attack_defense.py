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


class AttackDefenseClassifier:
    def __init__(
        self,
        baseline_model_path,
        dataset_root,
        attack="PGD",
        pgd_params=PGDParams(0.01, 0.001, 250),  # epsilon, step_size, num_iterations
        ddn_params=DDNParams(0.001, 0.05, 300),  # alpha, gamma, num_iterations
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
                split=[0.8, 0.2],
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
        data_denorm = denorm(img)

        # perform attack
        if self.attack == "PGD":
            perturbed_img = pgd_attack(
                model,
                data_denorm,
                label,
                self.pgd_params.epsilon,
                self.pgd_params.step_size,
                self.pgd_params.num_iterations,
            )
            perturbed_img = norm(perturbed_img)
        elif self.attack == "DDN":
            perturbed_data, success, _, last_perturbed_data = ddn_attack(
                model,
                data_denorm,
                label,
                self.ddn_params.num_iterations,
                self.ddn_params.alpha,
                self.ddn_params.gamma,
            )

            if success:
                perturbed_img = last_perturbed_data
            else:
                perturbed_img = perturbed_data
            perturbed_img = norm(perturbed_img)

        return (True, perturbed_img)

    def _resample_img(self, img, scale_factor=0.5):
        new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
        img = img.resize(new_size, Image.BILINEAR)
        img = img.resize((img.width * 2, img.height * 2), Image.BILINEAR)

        return img

    def _tensor_to_pil(self, tensor):
        return ToPILImage()(denorm(tensor).detach().cpu().squeeze())

    def _defend_feat_squeezing(self, model, perturbed_img, initial_pred, thresh=1.0):
        """Defends model with Feature squeezing. Return False if no attack is detected and true if attack is detected"""
        # median filter and resampling for feature squeezing
        perturbed_pil = self._tensor_to_pil(perturbed_img)
        median_img = perturbed_pil.filter(ImageFilter.MedianFilter(size=3))
        resampled_img = self._resample_img(perturbed_pil)

        # apply val transformations
        median_tesnor = val_tfs(median_img).unsqueeze(0).to(self.device)
        resampled_tensor = val_tfs(resampled_img).unsqueeze(0).to(self.device)

        # inference and softmax
        median_out = (
            nn.functional.softmax(model(median_tesnor)).detach().cpu().numpy().squeeze()
        )
        resampled_out = (
            nn.functional.softmax(model(resampled_tensor))
            .detach()
            .cpu()
            .numpy()
            .squeeze()
        )

        # compute L1-norm
        init_pred_softmax = (
            nn.functional.softmax(initial_pred).detach().cpu().numpy().squeeze()
        )
        median_l1 = np.linalg.norm(init_pred_softmax - median_out, ord=1)
        resampled_l1 = np.linalg.norm(init_pred_softmax - resampled_out, ord=1)
        max_l1 = max(median_l1, resampled_l1)

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

    def _compute_metrics(
        self,
        total_attacks: int,
        correct_after_attack: int,
        defence_metr=Union[ConfusionMatrix, int],
    ):
        """Dumps metrics in .json file for Feature Squeezing"""
        results = {}
        attack_metrics = {}
        defense_metrics = {}

        results["total_samples"] = len(self.val_dataloader)
        results["total_attacks"] = total_attacks

        # attack metrics
        attack_name = self._get_attack_filename()
        attack_metrics["attack"] = attack_name
        attack_metrics["correct_after_attack"] = correct_after_attack
        attack_metrics["model_accuracy"] = correct_after_attack / total_attacks
        attack_metrics["comment"] = (
            "This represents the number of samples correctly classified by the model after attack."
        )

        if self.defence == "Feat_squeezing" and isinstance(
            defence_metr, ConfusionMatrix
        ):
            defense_name = self.defence
            # compute accuracy, precision and recall for defense
            cm = defence_metr
            total = cm.tp + cm.tn + cm.fp + cm.fn
            accuracy = (cm.tp + cm.tn) / total if total > 0 else 0
            precision = cm.tp / (cm.tp + cm.fp) if (cm.tp + cm.fp) > 0 else 0
            recall = cm.tp / (cm.tp + cm.fn) if (cm.tp + cm.fn) > 0 else 0
            defense_metrics["defence"] = self.defence
            defense_metrics["TP_defense"] = cm.tp
            defense_metrics["TN_defense"] = cm.tn
            defense_metrics["FP_defense"] = cm.fp
            defense_metrics["FN_defense"] = cm.fn
            defense_metrics["accuracy"] = accuracy
            defense_metrics["precision"] = precision
            defense_metrics["recall"] = recall
            defense_metrics["comment"] = "Attack detection metrics"
        elif self.defence == "Distillation" and isinstance(defence_metr, int):
            defense_name = f"{self.defence}_temp{self.distillation_temp}"
            defense_metrics["defence"] = self.defence
            defense_metrics["correct after defense"] = defence_metr
            defense_metrics["model_accuracy"] = defence_metr / total_attacks
            defense_metrics["comment"] = "Model classification accuracy after defense."

        results["attack_metrics"] = attack_metrics
        results["defence_metrics"] = defense_metrics

        filename = attack_name + "_" + defense_name + ".json"
        out_file = os.path.join(self.out_dir, filename)
        with open(out_file, "w") as json_file:
            json.dump(results, json_file, indent=4)

    def attack_defend(self):
        # total number of attacks
        attacks = 0

        # number of correctly predicted outputs after attack
        correct = 0
        correct_defense = 0
        # number of detected attacks by the defence
        detected_attacks = 0
        confusion_matrix = ConfusionMatrix()
        attack_successul_count = 0

        for sample in tqdm.tqdm(self.val_dataloader, desc="Validation set"):
            gt = sample["qry_gt"].item()
            # print(f"gt: {gt}")
            attack_successful = False

            can_attack, perturbed_img = self._attack(self.baseline_model, sample)

            # attack samples where model is right
            if can_attack:
                attacks += 1
                attacked_output = self.baseline_model(perturbed_img)
                attacked_pred = attacked_output.max(1, keepdim=True)[1].item()

                # check if the attack was successful
                if attacked_pred == gt:
                    correct += 1
                    attack_successful = False
                else:
                    attack_successul_count += 1
                    attack_successful = True
            else:
                continue

            if self.defence == "Feat_squeezing":
                attack_detected = self._defend_feat_squeezing(
                    self.baseline_model, perturbed_img, attacked_output
                )
                confusion_matrix.update(attack_successful, attack_detected)

            elif self.defence == "Distillation":
                pred = self._defend_distillation(self.distillation_model, perturbed_img)

                if pred == gt:
                    correct_defense += 1

        if self.defence == "Feat_squeezing":
            self._compute_metrics(attacks, correct, confusion_matrix)
        elif self.defence == "Distillation":
            self._compute_metrics(attacks, correct, correct_defense)

        print(f"Number of attacks: {attacks}")
        print(
            f"Number of correct samples after {self.attack} attack (no defence): {correct}"
        )
        print(
            f"Number of correct samples after {self.defence} defence: {correct_defense}"
        )
        print(f"Successsful attacks: {attack_successul_count}")

        # print(f"defense fp: {confusion_matrix.fp}")
        # print(f"defense tp: {confusion_matrix.tp}")
        # print(f"defense tn: {confusion_matrix.tn}")
        # print(f"defense fn: {confusion_matrix.fn}")


if __name__ == "__main__":
    model = VGG10_lighter(num_classes=10)
    pretrained_path = "checkpoints/VGG10lightweight_10epchs1e-4_5epochs1e-5.pth"
    dataset_root = "/home/ModelRobustnessClassifier/dataset/raw-img"

    # pgd_hyperparameters
    pgd_epsilon = [0.01, 0.02, 0.03, 0.04, 0.05]
    pgd_iter = 250
    pgd_step_size = 0.001

    # attack with pgd
    # for eps in pgd_epsilon:
    #     pgd_params = PGDParams(eps, pgd_step_size, pgd_iter)
    #     # defence: Feature Squeezing
    #     processor_sqz = AttackDefenseClassifier(
    #         pretrained_path,
    #         dataset_root,
    #         attack="PGD",
    #         pgd_params=pgd_params,
    #         defence="Feat_squeezing",
    #     )
    #     processor_sqz.attack_defend()

    pgd_params = PGDParams(pgd_epsilon[1], pgd_step_size, pgd_iter)
    processor = AttackDefenseClassifier(
        pretrained_path,
        dataset_root,
        attack="PGD",
        defence="Feat_squeezing",
        pgd_params=pgd_params,
        distillation_model="/home/ModelRobustnessClassifier/src/checkpoints/models_temp_1/model_14.pth",
    )
    processor.attack_defend()
