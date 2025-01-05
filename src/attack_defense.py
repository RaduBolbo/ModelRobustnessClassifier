from networks.vgg10 import VGG10_lighter
from dataloaders import get_dls
from hyperparameters import *
from pgd_attack import pgd_attack, denorm

import torch
import os
import tqdm

torch.manual_seed(2024)

ATTACKS = ["PGD", "FGSM"]
DEFENECES = ["Distillation", "Feat_squeezing", None]


class AttackDefenseClassifier:
    def __init__(
        self,
        baseline_model_path,
        dataset_root,
        attack="PGD",
        defence=None,
        device="cuda",
    ):
        self.device = device
        self.baseline_model = self._load_model(baseline_model_path)
        self.val_dataloader = self._load_val_dataloader(dataset_root)

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
        # pgd attack parameters
        epsilon = 0.001
        step_size = 0.0001
        num_iterations = 100

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

    def _defend_feat_squeezing(self, model, sample, initial_pred):
        """Defends model with Feature squeezing. Return False if no attack is detected and true if attack is detected"""
        median_filter_img = sample["median_filter_img"].to(self.device)
        median_out = model(median_filter_img)
        median_pred = median_out.max(1, keepdim=True)[1].item()

        resampled_img = sample["resampled_img"].to(self.device)
        resampled_out = model(resampled_img)
        resampled_pred = resampled_out.max(1, keepdim=True)[1].item()

        # check if original prediction and squeezed predictions are the same
        if initial_pred == median_pred == resampled_pred:
            # no attack detected
            return False
        else:
            # attack detected
            return True

    def attack_defend(self):
        # total number of attacks
        attacks = 0

        # number of correctly predicted outputs after attack
        correct = 0

        # number of detected attacks by the defence
        detected_attacks = 0

        for sample in tqdm.tqdm(self.val_dataloader, desc="Validation set"):
            gt = sample["qry_gt"].item()

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
                else:
                    continue

            if self.defence == "Feat_squeezing":
                attack_detected = self._defend_feat_squeezing(
                    self.baseline_model, sample, attacked_pred
                )

                if attack_detected:
                    detected_attacks += 1

        print(f"Number of attacks: {attacks}")
        print(
            f"Number of correct samples after {self.attack} attack (no defence): {correct}"
        )
        print(
            f"Number of detected attacks after {self.defence} defence: {detected_attacks}"
        )


if __name__ == "__main__":
    model = VGG10_lighter(num_classes=10)
    pretrained_path = "checkpoints/VGG10lightweight_10epchs1e-4_5epochs1e-5.pth"
    dataset_root = "/home/ModelRobustnessClassifier/dataset/raw-img"

    attack_def_obj = AttackDefenseClassifier(
        pretrained_path, dataset_root, defence="Feat_squeezing"
    )

    attack_def_obj.attack_defend()
    # PGD attack, Feature Squeezing defence
    # Number of attacks: 4631
    # Number of correct samples after PGD attack (no defence): 4616
    # Number of detected attacks after Feat_squeezing defence: 1169
