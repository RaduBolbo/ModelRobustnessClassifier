'''
This is a script used to estimate the Lipschitz constant for the classifier
'''
from networks.vgg10 import VGG10_lighter
from torch.linalg import norm as l2_norm
import torch


pretraiend_path = r'checkpoints/VGG10lightweight_10epchs1e-4_5epochs1e-5.pth'
#pretraiend_path = r'checkpoints/checkpoints_distillation/checkpoints_distillation/student_temp20.pth'

model = VGG10_lighter(num_classes=10)
state_dict = torch.load(pretraiend_path)  # Load the state dictionary from the .pth file
model.load_state_dict(state_dict)


def compute_spectral_norm_conv_naive(weight, num_iters=100): # **** choose num_iters high enough so that the value dosn't vary between different runs
    """
    Compute the spectral norm of a Conv2d layer weight using power iteration.
    """
    out_channels, in_channels, k_h, k_w = weight.shape
    weight_matrix = weight.reshape(out_channels, -1)
    u = torch.randn(out_channels, device=weight.device)
    u = u / torch.norm(u)  # Normalize
    for _ in range(num_iters):
        v = torch.matmul(weight_matrix.T, u)
        v = v / torch.norm(v)
        u = torch.matmul(weight_matrix, v)
        u = u / torch.norm(u)
    return torch.dot(u, torch.matmul(weight_matrix, v)).item()

def compute_spectral_norm_conv(weight, num_iters=100):
    '''
    Impleemntin Gram from "Efficient Bound of Lipschitz Constant for Convolutional Layers by Gram Iteration" (2305.16173)
    '''
    weight_fft = torch.fft.fft2(weight, norm='ortho')

    D = weight_fft.permute(2, 3, 0, 1).reshape(-1, weight.shape[0], weight.shape[1])

    r = torch.zeros(D.shape[0], device=weight.device) 
    for _ in range(num_iters):
        norm_D = torch.linalg.norm(D, ord='fro', dim=(1, 2), keepdim=True)
        r += 2 * torch.log(norm_D.squeeze())
        D = D / norm_D 
        D = torch.matmul(D.transpose(-1, -2), D) 

    norm_D = torch.linalg.norm(D, ord='fro', dim=(1, 2), keepdim=True)
    spectral_norm = torch.max(norm_D * torch.exp(r.unsqueeze(-1) * 2 ** (-num_iters))).item()

    return spectral_norm

def compute_spectral_norm_dense(weight, num_iters=100):
    """
    Compute the spectral norm of a dense weight matrix using power iteration.
    """
    u = torch.randn(weight.size(0), device=weight.device)
    u = u / torch.norm(u)
    
    for _ in range(num_iters):
        v = torch.matmul(weight.T, u)
        v = v / torch.norm(v)
        u = torch.matmul(weight, v)
        u = u / torch.norm(u)
    
    sigma = torch.dot(u, torch.matmul(weight, v))  # Approximate largest singular value
    return sigma.item()

def compute_layer_lipschitz(layer, distribution_sigma):
    """
    Compute the Lipschitz constant for a given layer.
    """
    if isinstance(layer, torch.nn.Conv2d):
        return compute_spectral_norm_conv(layer.weight)
    elif isinstance(layer, torch.nn.Linear):
        #return l2_norm(layer.weight, ord=2).item() 
        return compute_spectral_norm_dense(layer.weight)
    elif isinstance(layer, torch.nn.BatchNorm2d):
        # Assume `sigma` is provided as a parameter (average_sigma of the dataset)
        # Compute gamma / sigma
        #return 1
        gamma = layer.weight.abs().max().item()  # Absolute value of gamma
        return gamma / distribution_sigma  # Divide by sigma
    
    elif isinstance(layer, (torch.nn.ReLU, torch.nn.MaxPool2d)):
        return 1.0
    elif isinstance(layer, torch.nn.PReLU):
        return layer.weight.abs().max().item()  # **** TO INVESTIGATE
        #return 1
    else:
        return 1.0

def compute_network_lipschitz(model, distribution_sigma):
    """
    Compute the Lipschitz constant for the entire network as the product
    of layer-wise Lipschitz constants.
    """
    lipschitz_constant = 1.0
    for name, module in model.named_modules():
        layer_lipschitz = compute_layer_lipschitz(module, distribution_sigma)
        lipschitz_constant *= layer_lipschitz
        print(f"Layer: {name}, Lipschitz constant: {layer_lipschitz}")
    return lipschitz_constant


# Compute the Lipschitz constant
distribution_sigma = 1 # it is not unique. It should be compute dfor each tensor at the input of BatchNorm
total_lipschitz_constant = compute_network_lipschitz(model, distribution_sigma)
print(f"Estimated Lipschitz constant for the network: {total_lipschitz_constant}")

"""
# V1) It gives very very small values
def compute_spectral_norm(layer):
    '''
    Returns the spectral norm of a layer
    '''
    if hasattr(layer, 'weight') and layer.weight is not None:
        weight = layer.weight.data
        if weight.ndim < 2:
            return 1.0
        spectral_norm_val = torch.linalg.svd(weight, full_matrices=False)[1].max()
        return spectral_norm_val.item()
    return 1.0  # nonlinear layers -> 1 (Ex: ReLU, MaxPool)

 
lipschitz_constant = 1.0 # initial Lipschitz constant
for name, module in model.named_modules(): # iterate over the nn's layers
    spectral_norm = compute_spectral_norm(module)
    lipschitz_constant *= spectral_norm
    print(f"Layer: {name} has spectral norm: {spectral_norm}")

print(f"Estimated Lipschitz Constant (spectral norm): {lipschitz_constant}")
"""