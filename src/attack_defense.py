from dataclasses import dataclass
from networks.vgg10 import VGG10_lighter
from dataloaders import get_dls
from hyperparameters import *
from pgd_attack import pgd_attack, denorm
import json

import torch
import os
import tqdm

torch.manual_seed(2024)

ATTACKS = ["PGD", "FGSM"]
DEFENECES = ["Distillation", "Feat_squeezing", None]


@dataclass
class PGDParams:
    epsilon: float
    step_size: float
    num_iterations: int


@dataclass
class ConfusionMatrix:
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0


class AttackDefenseClassifier:
    def __init__(
        self,
        baseline_model_path,
        dataset_root,
        attack="PGD",
        pgd_params=PGDParams(0.01, 0.001, 250),  # epsilon, step_size, num_iterations
        defence=None,
        device="cuda",
        out_dir="results/",
    ):
        self.device = device
        self.baseline_model = self._load_model(baseline_model_path)
        self.val_dataloader = self._load_val_dataloader(dataset_root)
        self.pgd_params = pgd_params
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

    def _attack_pgd(self, model, sample):
        """Perturbs the input with PGD."""

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
            model,
            data_denorm,
            label,
            self.pgd_params.epsilon,
            self.pgd_params.step_size,
            self.pgd_params.num_iterations,
        )

        # renormalize, because the image was denormalized and now it has to be sent to the nn again
        perturbed_data_normalized = T.Normalize(mean=mean, std=std)(perturbed_data)

        return (True, perturbed_data_normalized)

    def _defend_feat_squeezing(self, model, sample, initial_pred):
        """Defends model with Feature squeezing. Return False if no attack is detected and true if attack is detected"""
        median_filter_img = sample["median_filter_img"].to(self.device)
        median_out = model(median_filter_img)
        median_pred = median_out.max(1, keepdim=True)[1].item()

        resampled_img = sample["resampled_img"].to(self.device)
        resampled_out = model(resampled_img)
        resampled_pred = resampled_out.max(1, keepdim=True)[1].item()

        # check if original prediction and squeezed predictions are the same
        if initial_pred == median_pred or initial_pred == resampled_pred:
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

    def _dump_metrics_squeezing(
        self, cm: ConfusionMatrix, total_attacks: int, correct_after_attack: int
    ):
        """Dumps metrics in .json file for Feature Squeezing"""
        # create filename
        if self.attack == "PGD":
            filename = f"{self.attack}_eps{self.pgd_params.epsilon}_step{self.pgd_params.step_size}_iter{self.pgd_params.num_iterations}"
        filename = filename + f"_{self.defence}.json"
        out_file = os.path.join(self.out_dir, filename)

        results_dict = {}
        results_dict["total_samples"] = len(self.val_dataloader)
        results_dict["total_attacks"] = total_attacks
        results_dict["correct_after_attack"] = correct_after_attack
        results_dict["TP_defense"] = cm.tp
        results_dict["TN_defense"] = cm.tn
        results_dict["FP_defense"] = cm.fp
        results_dict["FN_defense"] = cm.fn

        total = cm.tp + cm.tn + cm.fp + cm.fn
        accuracy = (cm.tp + cm.tn) / total if total > 0 else 0
        precision = cm.tp / (cm.tp + cm.fp) if (cm.tp + cm.fp) > 0 else 0
        recall = cm.tp / (cm.tp + cm.fn) if (cm.tp + cm.fn) > 0 else 0

        results_dict["accuracy"] = accuracy
        results_dict["precision"] = precision
        results_dict["recall"] = recall

        with open(out_file, "w") as json_file:
            json.dump(results_dict, json_file, indent=4)

    def attack_defend(self):
        # total number of attacks
        attacks = 0

        # number of correctly predicted outputs after attack
        correct = 0
        correct_defense = 0
        # number of detected attacks by the defence
        detected_attacks = 0
        confusion_matrix = ConfusionMatrix()

        for sample in tqdm.tqdm(self.val_dataloader, desc="Validation set"):
            gt = sample["qry_gt"].item()
            # print(f"gt: {gt}")
            attack_successful = False

            if self.attack == "PGD":
                can_attack, perturbed_img = self._attack_pgd(
                    self.baseline_model, sample
                )

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
                        attack_successful = True
                else:
                    continue

            if self.defence == "Feat_squeezing":
                attack_detected = self._defend_feat_squeezing(
                    self.baseline_model, sample, attacked_pred
                )

                if attack_detected and attack_successful:
                    # attack correctly detected
                    confusion_matrix.tp += 1
                elif attack_detected and not attack_successful:
                    # input incorrectly flagged as attack
                    confusion_matrix.fp += 1  #
                elif not attack_detected and not attack_successful:
                    # input correctly flagged as no attack
                    confusion_matrix.tn += 1
                elif not attack_detected and attack_successful:
                    # input incorrectly flagged as no attack
                    confusion_matrix.fn += 1
            elif self.defence == "Distillation":
                model_path = "/home/ModelRobustnessClassifier/src/checkpoints/models_temp_5/model_14.pth"
                pred = self._defend_distillation(model_path, perturbed_img)

                if pred == gt:
                    correct_defense += 1

        if self.defence == "Feat_squeezing":
            self._dump_metrics_squeezing(confusion_matrix, attacks, correct)

        print(f"Number of attacks: {attacks}")
        print(
            f"Number of correct samples after {self.attack} attack (no defence): {correct}"
        )
        print(
            f"Number of detected attacks after {self.defence} defence: {detected_attacks}"
        )
        print(f"Number of ocrrect samples after attack (defence): {correct_defense}")

        print(f"defense fp: {confusion_matrix.fp}")
        print(f"defense tp: {confusion_matrix.tp}")
        print(f"defense tn: {confusion_matrix.tn}")
        print(f"defense fn: {confusion_matrix.fn}")


if __name__ == "__main__":
    model = VGG10_lighter(num_classes=10)
    pretrained_path = "checkpoints/VGG10lightweight_10epchs1e-4_5epochs1e-5.pth"
    dataset_root = "/home/ModelRobustnessClassifier/dataset/raw-img"

    # pgd_hyperparameters
    pgd_epsilon = [0.01, 0.02, 0.03, 0.04, 0.05]
    pgd_iter = 250
    pgd_step_size = 0.0001

    attack_def_obj = AttackDefenseClassifier(
        pretrained_path, dataset_root, defence="Feat_squeezing"
    )

    attack_def_obj.attack_defend()
    # PGD attack, Feature Squeezing defence
    # Number of attacks: 4631
    # Number of correct samples after PGD attack (no defence): 4616
    # Number of detected attacks after Feat_squeezing defence: 1169
