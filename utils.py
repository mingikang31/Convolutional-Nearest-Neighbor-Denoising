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
            img_to_show = img_tensor.squeeze(0).numpy()
            ax.imshow(img_to_show, cmap='gray', vmin=0, vmax=1)
        elif img_tensor.shape[0] == 3:
            # RGB image: permute channels for plotting -> [H, W, C]
            img_to_show = img_tensor.permute(1, 2, 0)
            # Clamp to [0,1] range for display
            img_to_show = torch.clamp(img_to_show, 0, 1).numpy()
            ax.imshow(img_to_show)
        
        ax.set_title(title, fontsize=12)
        ax.axis('off') # Hide axes ticks

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_denoising_results_1D(clean_signals, noisy_signals, denoised_signals, labels, save_path, 
                                   title="1D Signal Denoising Results", ratio=2.6, dark_mode=False, zoom=1):
    """
    Visualize 1D signal denoising results in MNIST1D format.
    
    Args:
        clean_signals: Clean ground truth signals (tensor, [N, C, T])
        noisy_signals: Noisy input signals (tensor, [N, C, T]) 
        denoised_signals: Denoised output signals (tensor, [N, C, T])
        labels: Class labels for each signal (tensor, [N])
        save_path: Path to save the visualization
        title: Title for the plot
        ratio: Height ratio for subplots
        dark_mode: Use dark theme if True
        zoom: Zoom level for plot limits
    """
    import matplotlib.pyplot as plt
    from utils import measure_psnr
    
    # Move to CPU and detach from computation graph
    clean_signals = clean_signals.detach().cpu()
    noisy_signals = noisy_signals.detach().cpu()
    denoised_signals = denoised_signals.detach().cpu()
    labels = labels.detach().cpu() if hasattr(labels, 'detach') else labels
    
    # Get number of signals to plot (max 10)
    n_signals = min(len(clean_signals), 10)
    
    # Create time vector (assuming normalized range like original MNIST1D)
    seq_len = clean_signals.shape[-1]
    t = torch.linspace(-5, 5, seq_len) / 6.0  # Following original MNIST1D scaling
    
    # Create figure with 3 rows (clean, noisy, denoised) and n_signals columns
    rows, cols = 3, n_signals
    fig = plt.figure(figsize=[cols*1.5, rows*1.5*ratio], dpi=60)
    
    for c in range(cols):
        # Extract signals (squeeze channel dimension if present)
        clean_x = clean_signals[c].squeeze() if clean_signals[c].ndim > 1 else clean_signals[c]
        noisy_x = noisy_signals[c].squeeze() if noisy_signals[c].ndim > 1 else noisy_signals[c]
        denoised_x = denoised_signals[c].squeeze() if denoised_signals[c].ndim > 1 else denoised_signals[c]
        
        # Calculate PSNR values
        psnr_noisy = measure_psnr(clean_signals[c:c+1], noisy_signals[c:c+1]).item()
        psnr_denoised = measure_psnr(clean_signals[c:c+1], denoised_signals[c:c+1]).item()
        
        # Plot clean signal (top row)
        ax1 = plt.subplot(rows, cols, c + 1)
        if dark_mode:
            plt.plot(clean_x, t, 'wo', linewidth=6)
            ax1.set_facecolor('k')
        else:
            plt.plot(clean_x, t, 'k-', linewidth=2)
        
        plt.title(f"Clean\nLabel: {int(labels[c])}", fontsize=12)
        plt.xlim(-zoom, zoom)
        plt.ylim(-zoom, zoom)
        plt.gca().invert_yaxis()
        plt.xticks([])
        plt.yticks([])
        
        # Plot noisy signal (middle row)
        ax2 = plt.subplot(rows, cols, cols + c + 1)
        if dark_mode:
            plt.plot(noisy_x, t, 'ro', linewidth=6)
            ax2.set_facecolor('k')
        else:
            plt.plot(noisy_x, t, 'r-', linewidth=2)
        
        plt.title(f"Noisy\nPSNR: {psnr_noisy:.1f}dB", fontsize=12)
        plt.xlim(-zoom, zoom)
        plt.ylim(-zoom, zoom)
        plt.gca().invert_yaxis()
        plt.xticks([])
        plt.yticks([])
        
        # Plot denoised signal (bottom row)
        ax3 = plt.subplot(rows, cols, 2*cols + c + 1)
        if dark_mode:
            plt.plot(denoised_x, t, 'go', linewidth=6)
            ax3.set_facecolor('k')
        else:
            plt.plot(denoised_x, t, 'g-', linewidth=2)
        
        plt.title(f"Denoised\nPSNR: {psnr_denoised:.1f}dB", fontsize=12)
        plt.xlim(-zoom, zoom)
        plt.ylim(-zoom, zoom)
        plt.gca().invert_yaxis()
        plt.xticks([])
        plt.yticks([])
    
    # Add overall title
    fig.suptitle(title, fontsize=16, y=0.98)
    
    # Adjust layout and save
    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()