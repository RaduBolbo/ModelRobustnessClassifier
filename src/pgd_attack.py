'''
PGD inspired by FGSM in https://pytorch.org/tutorials/beginner/fgsm_tutorial.html
'''

import torch
import torch.nn.functional as F
from networks.vgg10 import VGG10, VGG10_lighter

import os, torch, random, shutil, numpy as np, pandas as pd
from glob import glob; from PIL import Image
Image.MAX_IMAGE_PIXELS = None 
from torchvision import transforms as T
import torchvision
torch.manual_seed(2024)
from dataloaders import get_dls

from tqdm import tqdm
import pickle
import torchattacks

device = 'cuda'

# def pgd_attack(image, data_grad, epsilon, step_size, num_iterations=40):
#     # 1) clone the orignal image
#     perturbed_image = image.clone().detach()

#     # 2) add noise iteratively
#     for _ in range(num_iterations):
#         # a) add the noise (this is almost FGSM)
#         perturbed_image = perturbed_image + step_size * data_grad.sign()
#         # b) reproject if the perturbation to be at maximum of epsilon magnitude (clamp is element-wise so that works)
#         perturbed_image = torch.clamp(perturbed_image, image - epsilon, image + epsilon)
#         # c) plamp again to ensure this is not outside the image value range.
#         perturbed_image = torch.clamp(perturbed_image, 0, 1)

#     return perturbed_image

def serialize_adv_examples(adv_examples, output_path):
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(adv_examples, f)
        print(f"Adversarial examples successfully serialized to {output_path}")
    except Exception as e:
        print(f"Failed to serialize adversarial examples: {e}")


def pgd_attack(model, image, target, epsilon, step_size, num_iterations=40):
    model.eval()
    # 1) clone the original image
    perturbed_image = image.clone().detach().requires_grad_(True)

    # 2) add noise iteratively
    for _ in range(num_iterations):

        # determine the data gradient because its sign is needed to geenrate the noise
        output = model(perturbed_image)
        loss = F.nll_loss(output, target)
        model.zero_grad()
        loss.backward()
        data_grad = perturbed_image.grad.data

        # a) add the noise (this is almost FGSM)
        perturbed_image = perturbed_image + step_size * data_grad.sign()

        # b) reproject if the perturbation to be at maximum of epsilon magnitude (clamp is element-wise so that works)
        perturbed_image = torch.clamp(perturbed_image, image - epsilon, image + epsilon)

        # c) clamp again to ensure this is not outside the image value range. # but... th eimage range is not [0, 1] so don't apply 
        #perturbed_image = torch.clamp(perturbed_image, 0, 1)

        perturbed_image = perturbed_image.clone().detach().requires_grad_(True) # re-enable gradient requires because I need them for the next iteration

    return perturbed_image

def denorm(batch, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225], size = 224):
    # **** the body mush be rewritten because we have another datalaoder
    if isinstance(mean, list):
        mean = torch.tensor(mean).to(device)
    if isinstance(std, list):
        std = torch.tensor(std).to(device)

    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)

def test_pgd(model, pretraiend_path, device, epsilon, step_size, num_iterations):
    batch_size = 1
    num_workers = 1

    model = model.to(device)
    if pretraiend_path is not None:
        state_dict = torch.load(pretraiend_path)  # Load the state dictionary from the .pth file
        model.load_state_dict(state_dict)

    root = "dataset/raw-img"
    mean, std, size = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 224 # media, deviatai standard, diemsniuenaimaginilor
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
    #tr_dl, val_dl, classes, cls_counts = get_dls(root = root, train_transformations = train_tfs, val_transformations = val_tfs, batch_size = batch_size, split = [0.8, 0.2], num_workers = num_workers)
    tr_dl, val_dl, classes, cls_counts = get_dls(root = root, train_transformations = train_tfs, val_transformations = val_tfs, batch_size = batch_size, split = [0.995, 0.01], num_workers = num_workers)
    test_loader = val_dl

    correct = 0
    attacked = 0
    adv_examples = []

    for batch in tqdm(test_loader, desc="Validation"):
        data = batch["qry_im"].to(device)
        target = batch["qry_gt"].to(device)

        data.requires_grad = True

        # forward pass
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the prediction

        # don't attack if label is already wrong
        if init_pred.item() != target.item():
            continue

        # denormalize
        #data = denorm(data)

        # perform PGD
        perturbed_data = pgd_attack(model, data, target, epsilon, step_size, num_iterations)

        # renormalize, becaude the image was denormalized and now it has to be sent to the nn again
        #perturbed_data = T.Normalize(mean=mean, std=std)(perturbed_data)

        # reclassifyu to see if it is now wrong
        output = model(perturbed_data)

        # check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        attacked += 1
        if final_pred.item() == target.item():
            correct += 1
            #print('correct')
            # special case for saving 0 epsilon examples
            if epsilon == 0 and len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            #print('wrong')
            # save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
            # print('perturbed_data ', torch.max(perturbed_data))
            # print('data ', torch.max(data))
            # data = denorm(data)
            # perturbed_data = denorm(perturbed_data)
            # print(final_pred.item(), target.item())
            # torchvision.utils.save_image(data, 'data.png')
            # torchvision.utils.save_image(perturbed_data, 'perturbed_data.png')
            # torchvision.utils.save_image((data-perturbed_data)*250, 'dif.png')

    final_acc = correct/float(attacked)
    print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {attacked} = {final_acc}")

    return final_acc, adv_examples

def test_pgd_with_torchattacks(model, pretrained_path, device, epsilon, step_size, num_iterations):
    batch_size = 1
    num_workers = 1

    model = model.to(device)
    if pretrained_path is not None:
        state_dict = torch.load(pretrained_path)  # Load the state dictionary from the .pth file
        model.load_state_dict(state_dict)

    root = "dataset/raw-img"
    mean, std, size = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 224 # media, deviatai standard, diemsniuenaimaginilor
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
    #tr_dl, val_dl, classes, cls_counts = get_dls(root = root, train_transformations = train_tfs, val_transformations = val_tfs, batch_size = batch_size, split = [0.8, 0.2], num_workers = num_workers)
    tr_dl, val_dl, classes, cls_counts = get_dls(root = root, train_transformations = train_tfs, val_transformations = val_tfs, batch_size = batch_size, split = [0.99, 0.01], num_workers = num_workers)

    correct = 0
    attacked = 0
    adv_examples = []
    model.eval()

    attack = torchattacks.PGD(model, eps=epsilon, alpha=step_size, steps=num_iterations)
    
    correct = 0
    attacked = 0
    adv_examples = []

    for batch in tqdm(val_dl, desc="Validation"):
        data = batch["qry_im"].to(device)
        target = batch["qry_gt"].to(device)

        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]
        if init_pred.item() != target.item():
            continue

        perturbed_data = attack(data, target)

        output = model(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1]

        attacked += 1
        if final_pred.item() == target.item():
            correct += 1
            if epsilon == 0 and len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
        else:
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

    final_acc = correct / float(attacked)
    print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {attacked} = {final_acc}")
    return final_acc, adv_examples

if __name__ == '__main__':

    #model = VGG10(num_classes=10)
    #pretrained_path = 'checkpoints/VGG10.pth'

    model = VGG10_lighter(num_classes=10)
    pretrained_path = 'checkpoints/VGG10lightweight_10epchs1e-4_5epochs1e-5.pth'
    #pretrained_path = 'checkpoints/GOOD.pth'


    device = 'cuda'
    # set the attack parameters
    # Observations:
    # Obs 1) epsilon = 0.1; step_size = 0.0001; num_iterations = 25 => the classifier begins to have both correct and wrong parameters (maybe epsilon could be lowered anyway to 0.01 or 0.001)
    epsilon = 0.001 # 0.001 correct; 
    step_size = 0.001
    num_iterations = 100

    #final_acc, adv_examples = test_pgd(model, pretrained_path, device, epsilon, step_size, num_iterations)

    #final_acc, adv_examples = test_pgd(model, pretrained_path, device, 0.001, step_size, num_iterations)
    
    #epsilons = [0.00010, 0.00025, 0.00050, 0.00075, 0.00100, 0.002500, 0.00500, 0.00750, 0.001] # **** this may change if somethiong bad is observed
    
    implementation = 'official'
    if implementation == 'ours':
        epsilons = [0.0025, 0.0050, 0.0075, 0.0100, 0.01250, 0.01500, 0.01750] # **** this may change if somethiong bad is observed
        for epsilon in epsilons:
            final_acc, adv_examples = test_pgd(model, pretrained_path, device, epsilon, step_size, num_iterations)
            adv_examples_output_path = f'adv_examples/pgd/epsilon={epsilon}_step_size={step_size}_num_iterations={num_iterations}_final_acc={final_acc}.pkl'
            print(f'Final_ACC = {final_acc} for epsilon = {epsilon}')
            serialize_adv_examples(adv_examples, adv_examples_output_path)
    else:
        epsilons = [0.0025, 0.0050, 0.0075, 0.0100, 0.01250, 0.01500, 0.01750] # **** this may change if somethiong bad is observed
        for epsilon in epsilons:
            final_acc, adv_examples = test_pgd_with_torchattacks(model, pretrained_path, device, epsilon, step_size, num_iterations)
            adv_examples_output_path = f'adv_examples/pgd_official/epsilon={epsilon}_step_size={step_size}_num_iterations={num_iterations}_final_acc={final_acc}.pkl'
            print(f'Final_ACC = {final_acc} for epsilon = {epsilon}')
            serialize_adv_examples(adv_examples, adv_examples_output_path)


