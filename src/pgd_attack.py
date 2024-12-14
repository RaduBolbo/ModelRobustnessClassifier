import torch
import torch.nn.functional as F


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

def pgd_attack(model, image, target, epsilon, step_size, num_iterations=40):
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

        # c) clamp again to ensure this is not outside the image value range.
        perturbed_image = torch.clamp(perturbed_image, 0, 1)

        perturbed_image = perturbed_image.clone().detach().requires_grad_(True) # re-enable gradient requires because I need them for the next iteration

    return perturbed_image

def denorm(batch, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225], size = 224):
    # **** the body mush be rewritten because we have another datalaoder
    if isinstance(mean, list):
        mean = torch.tensor(mean).to(device)
    if isinstance(std, list):
        std = torch.tensor(std).to(device)

    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)

def test(model, device, test_loader, epsilon, step_size, num_iterations):
    # Accuracy counter
    correct = 0
    adv_examples = []

    # Loop over all examples in test set
    for data, target in test_loader:

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        # If the initial prediction is wrong, don't bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        # Restore the data to its original scale
        data_denorm = denorm(data)

        # Call PGD Attack
        perturbed_data = pgd_attack(model, data_denorm, target, epsilon, step_size, num_iterations)

        # Reapply normalization
        perturbed_data_normalized = transforms.Normalize((0.1307,), (0.3081,))(perturbed_data)

        # Re-classify the perturbed image
        output = model(perturbed_data_normalized)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if epsilon == 0 and len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(test_loader)} = {final_acc}")

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples





