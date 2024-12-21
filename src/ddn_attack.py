import torch
import torch.nn.functional as F
from networks.vgg10 import VGG10, VGG10_lighter

import os, torch, random, shutil, numpy as np, pandas as pd
from glob import glob; from PIL import Image
Image.MAX_IMAGE_PIXELS = None 
from torchvision import transforms as T
torch.manual_seed(2024)
from dataloaders import get_dls

from tqdm import tqdm
import pickle

device = 'cuda'

# def ddn_attack(model, image, target, num_iterations=40, alpha=0.05, gamma=0.05):
#     perturbed_image = image.clone().detach().requires_grad_(True)
#     epsilon = torch.ones(image.size(0), device=device)  # Dynamic norm for each image in the batch

#     for _ in range(num_iterations):
#         # compute gradient
#         output = model(perturbed_image)
#         loss = F.cross_entropy(output, target)
#         model.zero_grad()
#         loss.backward()
#         grad = perturbed_image.grad.data # the gradient is computed with respect to the perturbed image, so that the direction gests updated at each step

#         # update perturbed image
#         direction = grad / (torch.norm(grad.view(grad.size(0), -1), dim=1, keepdim=True).view(-1, 1, 1, 1) + 1e-8)
#         perturbed_image = perturbed_image - alpha * direction

#         # normalize perturbation
#         perturbed_image = image + (perturbed_image - image).renorm(p=2, dim=0, maxnorm=epsilon.view(-1, 1, 1, 1)) # renormed so that adv example doesn't get past the admited vecinuty

#         # clip
#         perturbed_image = torch.clamp(perturbed_image, 0, 1).detach().requires_grad_(True)

#         # Check if adversarial, adjust epsilon
#         with torch.no_grad():
#             output = model(perturbed_image)
#             pred = output.max(1, keepdim=True)[1]
#             is_adv = pred != target

#         if is_adv:
#             epsilon = epsilon * (1 - gamma) # decrease norm
#         else:
#             epsilon = epsilon * (1 + gamma) # increase norm

#     return perturbed_image

# def ddn_attack(model, image, target, num_iterations=40, alpha=0.05, gamma=0.05):
#     # Initialize variables
#     delta = torch.zeros_like(image, requires_grad=True).to(device)  # Perturbation
#     epsilon = torch.ones(image.size(0), device=device)  # Dynamic norm for each image in the batch

#     for _ in range(num_iterations):
#         # Compute gradient with respect to the perturbed image
#         perturbed_image = image + delta
#         perturbed_image = perturbed_image.requires_grad_(True)
#         output = model(perturbed_image)
#         loss = F.cross_entropy(output, target)
#         model.zero_grad()
#         loss.backward()
#         grad = perturbed_image.grad.data

#         direction = grad / (torch.norm(grad.view(grad.size(0), -1), dim=1, keepdim=True).view(-1, 1, 1, 1) + 1e-8) # compute the normalised gradient
#         delta = delta - alpha * direction # towards gradient ascending

#         # renormed so that adv example doesn't get past the admited vecinuty epslion
#         delta = delta.renorm(p=2, dim=0, maxnorm=epsilon.view(-1, 1, 1, 1).item())  

#         # perturb image, then clip
#         perturbed_image = torch.clamp(image + delta, 0, 1)

#         with torch.no_grad():
#             output = model(perturbed_image)
#             pred = output.max(1, keepdim=True)[1]
#             is_adv = pred != target

#         if is_adv:
#             epsilon = epsilon * (1 - gamma) # decrease norm
#         else:
#             epsilon = epsilon * (1 + gamma) # increase norm

#     return perturbed_image

def ddn_attack(model, image, target, num_iterations, alpha, gamma):
    delta = torch.zeros_like(image).to(device)
    epsilon = 0.1 * torch.ones(image.size(0), device=device) 
    succes_status = False
    smallest_perturbation = 1000000

    for k in range(num_iterations):
        perturbed_image = (image + delta).detach().requires_grad_(True)
        output = model(perturbed_image)
        loss = F.cross_entropy(output, target)
        model.zero_grad()
        loss.backward()
        grad = perturbed_image.grad  
        #print(f'{k} loss: {loss}')

        # Normalize gradient and update delta
        grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=1, keepdim=True).view(-1, 1, 1, 1) + 1e-8 
        direction = grad / grad_norm # compute the normalised gradient
        delta = delta + alpha * direction # towards gradient ascending # + or -? *****

        # renormed so that adv example doesn't get past the admited vecinuty epslion
        #delta = delta.renorm(p=2, dim=0, maxnorm=epsilon.view(-1, 1, 1, 1).item())
        delta = delta.renorm(p=2, dim=0, maxnorm=epsilon.item())

        # perturb image, then clip
        perturbed_image = torch.clamp(image + delta, 0, 1)

        with torch.no_grad():
            output = model(perturbed_image)
            pred = output.max(1, keepdim=True)[1]
            #print('pred: ', pred)
            #print('target: ', target)
            is_adv = pred != target

        if is_adv:
            succes_status = True # the atack was succesfull
            if smallest_perturbation > torch.norm((image - perturbed_image).view(image.size(0), -1), dim=1).item():
                smallest_perturbation = torch.norm((image - perturbed_image).view(image.size(0), -1), dim=1).item()
            epsilon = epsilon * (1 - gamma) # decrease norm
        else:
            epsilon = epsilon * (1 + gamma) # increase norm

    return perturbed_image, succes_status, smallest_perturbation


def denorm(batch, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225], size = 224):
    # **** the body mush be rewritten because we have another datalaoder
    if isinstance(mean, list):
        mean = torch.tensor(mean).to(device)
    if isinstance(std, list):
        std = torch.tensor(std).to(device)

    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)

def test_ddn(model, pretrained_path, device, num_iterations=40, alpha=0.05, gamma=0.05):
    batch_size = 1
    num_workers = 1

    model = model.to(device)
    if pretrained_path is not None:
        state_dict = torch.load(pretrained_path)  # Load the state dictionary from the .pth file
        model.load_state_dict(state_dict)

    root = "dataset/raw-img"
    mean, std, size = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 224
    val_tfs = T.Compose([T.ToTensor(),
        T.Resize(size = (size, size),
        antialias = False),
        T.Normalize(mean = mean, std = std)])
    train_tfs = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=45),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
        T.ToTensor(),
        T.RandomResizedCrop(size=(224, 224), scale=(0.7, 1.2)),  
        T.Resize(size = (size, size),
        antialias = False),
        T.Normalize(mean=mean, std=std),
        ])
    tr_dl, val_dl, classes, cls_counts = get_dls(root = root, train_transformations = train_tfs, val_transformations = val_tfs, batch_size = batch_size, split = [0.8, 0.2], num_workers = num_workers)
    #tr_dl, val_dl, classes, cls_counts = get_dls(root = root, train_transformations = train_tfs, val_transformations = val_tfs, batch_size = batch_size, split = [0.95, 0.05], num_workers = num_workers)
    #tr_dl, val_dl, classes, cls_counts = get_dls(root = root, train_transformations = train_tfs, val_transformations = val_tfs, batch_size = batch_size, split = [0.9995, 0.0005], num_workers = num_workers)
    test_loader = val_dl

    correct = 0
    attacked = 0
    adv_norms = []

    for batch in tqdm(test_loader, desc="Validation"):
        data = batch["qry_im"].to(device)
        target = batch["qry_gt"].to(device)

        data.requires_grad = True

        # forward pass
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the prediction

        # there is no need to attack
        if init_pred.item() != target.item():
            continue

        # denorm
        data_denorm = denorm(data)
        # perform DDN attack
        perturbed_data, success_status, perturbation_norm = ddn_attack(model, data_denorm, target, num_iterations, alpha, gamma)
        # compute L2 norm of perturbation
        #perturbation_norm = torch.norm((data_denorm - perturbed_data).view(data.size(0), -1), dim=1)
        adv_norms.append(perturbation_norm)
        perturbed_data_normalized = T.Normalize(mean=mean, std=std)(perturbed_data)
        output = model(perturbed_data_normalized)

        final_pred = output.max(1, keepdim=True)[1]
        attacked += 1
        if not success_status:
            correct += 1

        # **** uncomment these to find the right hyperaparameters for the attack
        #print('perturbation_norm.item(): ', perturbation_norm)
        #print(success_status)
        #print('------')

    final_acc = correct / float(attacked) # this should be 0 if the K iterations were enough
    avg_norm = np.mean(adv_norms)
    median_norm = np.median(adv_norms)
    print(f"Test Accuracy = {correct} / {attacked} = {final_acc}")
    print(f"Average L2 Norm: {avg_norm}")
    print(f"Median L2 Norm: {median_norm}")

    return final_acc, avg_norm, median_norm

if __name__ == '__main__':

    #model = VGG10(num_classes=10)
    #pretrained_path = 'checkpoints/VGG10.pth'

    model = VGG10_lighter(num_classes=10)
    pretrained_path = 'checkpoints/VGG10lightweight_10epchs1e-4_5epochs1e-5.pth'
    #pretrained_path = 'checkpoints/GOOD.pth'


    device = 'cuda'

    num_iterations = 300
    alpha = 0.001
    gamma = 0.05

    final_acc, avg_norm, median_norm = test_ddn(model, pretrained_path, device, num_iterations, alpha, gamma)
    print('final_acc: ', final_acc)
    print('avg_norm: ', avg_norm)
    print('median_norm: ', median_norm)
    