from hyperparameters import *
from dataloaders import get_dls
from pgd_attack import pgd_attack, denorm
from networks.vgg10 import VGG10_lighter

import torch
from tqdm import tqdm


def perform_pgd(model, sample):
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


def feat_squeezing(model, model_path, dataset_root):
    batch_size = 1
    num_workers = 1

    model = model.to(device)
    if model_path is not None:
        state_dict = torch.load(
            model_path
        )  # Load the state dictionary from the .pth file
        model.load_state_dict(state_dict)

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
    attacked = 0

    for sample in tqdm(val_dl, desc="Validation"):
        can_attack, perturbed_img = perform_pgd(model, sample)

        if not can_attack:
            print("Can't attack for current sample, model is already wrong. Moving on.")
        else:
            output = model(perturbed_img)
            output_filtered = model(sample["median_filter_img"].to(device))
            output_resampled = model(sample["resampled_img"].to(device))

            # check for success
            # get the index of the max log-probability
            final_pred = output.max(1, keepdim=True)[1]
            pred_median_filter = output_filtered.max(1, keepdim=True)[1]
            pred_resampled_img = output_resampled.max(1, keepdim=True)[1]
            print(f"Perturbed input prediction: {final_pred.item()}")
            print(f"Median filter input prediction: {pred_median_filter.item()}")
            print(f"Resampled input prediction: {pred_resampled_img.item()}")
            print(f"GT label: {sample['qry_gt'].item()}")
            attacked += 1
            if final_pred.item() == sample["qry_gt"].item():
                print("Attack not successful!")
                correct += 1
            else:
                print("Attack successful!")

    final_acc = correct / float(attacked)
    print(final_acc)


if __name__ == "__main__":
    model = VGG10_lighter(num_classes=10)
    pretrained_path = "checkpoints/VGG10lightweight_10epchs1e-4_5epochs1e-5.pth"
    dataset_root = "/home/ModelRobustnessClassifier/dataset/raw-img"
    feat_squeezing(model, pretrained_path, dataset_root)
    print("Feature squeezing")
