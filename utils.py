"""Utility Functions for the Project"""
import torch.nn as nn
import torch
import os 
import matplotlib.pyplot as plt
from typing import Optional

def write_to_file(file_path, data):
    """
    Write data to a file in a readable format.

    Args:
        file_path (str): The path to the file.
        data: The data to write to the file (can be various types).
    """
    with open(file_path, 'w') as file:
        if isinstance(data, list):
            # For lists like train_eval_results
            for item in data:
                file.write(f"{item}\n")
        elif hasattr(data, '__dict__'):
            # For objects like args
            for key, value in vars(data).items():
                file.write(f"{key}: {value}\n")
        elif isinstance(data, nn.Module):
            # For PyTorch models
            file.write(str(data))
        else:
            # Default case
            file.write(str(data))
            file.write("\n")

def set_seed(seed):
    """
    Set the random seed for reproducibility.

    Args:
        seed (int): The seed value.
    """
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def measure_psnr(img, img2):
    """Computes the PSNR (Peak Signal-to-Noise Ratio) between two images."""
    mse = nn.MSELoss()(img, img2)
    if mse == 0:
        return float('inf')  # If no noise is present, PSNR is infinite
    max_pixel = 1.0  # Assuming the images are normalized between 0 and 1
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr

def visualize_denoising_results(clean_img, noisy_img, denoised_img, save_path):
    """
    Visualize denoising results.
    Args:
        clean_img: Clean ground truth image (tensor, [C, H, W])
        noisy_img: Noisy input image (tensor, [C, H, W]) 
        denoised_img: Denoised output image (tensor, [C, H, W])
        save_path: Path to save the visualization
    """
    # Move to CPU and detach from computation graph
    clean_img = clean_img.detach().cpu()
    noisy_img = noisy_img.detach().cpu()  
    denoised_img = denoised_img.detach().cpu()

    # Calculate PSNR values (keep data in [0,1] range for PSNR calculation)
    psnr_noisy = measure_psnr(clean_img, noisy_img)
    psnr_denoised = measure_psnr(clean_img, denoised_img)

    # Create a figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # List of images and titles for easy iteration
    images = [clean_img, noisy_img, denoised_img]
    titles = [
        'Clean Ground Truth',
        f'Noisy Input\nPSNR: {psnr_noisy:.2f} dB',
        f'Denoised Output\nPSNR: {psnr_denoised:.2f} dB'
    ]

    for ax, img_tensor, title in zip(axes, images, titles):
        # Check if the image is grayscale or RGB
        if img_tensor.shape[0] == 1:
            # Grayscale image: remove channel dim for plotting -> [H, W]
            img_to_show = img_tensor.squeeze(0)
            ax.imshow(img_to_show, cmap='gray', vmin=0, vmax=1)
        elif img_tensor.shape[0] == 3:
            # RGB image: permute channels for plotting -> [H, W, C]
            img_to_show = img_tensor.permute(1, 2, 0)
            # Clamp to [0,1] range for display
            img_to_show = torch.clamp(img_to_show, 0, 1)
            ax.imshow(img_to_show)
        
        ax.set_title(title, fontsize=12)
        ax.axis('off') # Hide axes ticks

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()