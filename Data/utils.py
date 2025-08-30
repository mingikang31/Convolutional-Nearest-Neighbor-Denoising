import torch
import torch.nn as nn 
import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms, datasets
# from mnist1d.data import make_dataset
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d

class NoisyBSD68(Dataset):
    def __init__(self, data_dir, target_count=5000, target_size=224, noise_std=0.3):
        super().__init__()
        self.target_count = target_count
        self.noise_std = noise_std

        self.original_images = self.load_images(data_dir)

        self.transform = transforms.Compose([
            transforms.RandomCrop(target_size), 
            transforms.RandomVerticalFlip(p=0.5), 
            transforms.RandomHorizontalFlip(p=0.5), 
            transforms.RandomChoice([
                transforms.RandomRotation(degrees=(-90, -90)),
                transforms.RandomRotation(degrees=(90, 90)),
                transforms.RandomRotation(degrees=(180, 180)),
            ])
        ])

    def __len__(self):
        return self.target_count

    def __getitem__(self, index): 
        base_img = random.choice(self.original_images)
        clean_target = self.transform(base_img)
        noise = torch.randn_like(clean_target) * self.noise_std 
        noisy_img = torch.clamp(clean_target + noise, 0., 1.)

        return noisy_img, clean_target

    @staticmethod 
    def load_images(data_dir):
        images = []
        for filename in os.listdir(data_dir):
            try:
                img = Image.open(os.path.join(data_dir, filename)).convert('L') # Grayscale
                img_np = np.array(img, dtype=np.float32) / 255.0
                # Convert to [C, H, W] tensor format
                img_torch = torch.from_numpy(img_np).unsqueeze(0) 
                images.append(img_torch)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        return images

class NoisyCBSD68(Dataset):
    def __init__(self, data_dir, target_count=5000, target_size=224, noise_std=0.3):
        super().__init__()
        self.target_count = target_count
        self.noise_std = noise_std

        self.original_images = self.load_images(data_dir)

        self.transform = transforms.Compose([
            transforms.RandomCrop(target_size), 
            transforms.RandomVerticalFlip(p=0.5), 
            transforms.RandomHorizontalFlip(p=0.5), 
            transforms.RandomChoice([
                transforms.RandomRotation(degrees=(-90, -90)),
                transforms.RandomRotation(degrees=(90, 90)),
                transforms.RandomRotation(degrees=(180, 180)),
            ])
        ])

    def __len__(self):
        return self.target_count

    def __getitem__(self, index): 
        base_img = random.choice(self.original_images)
        clean_target = self.transform(base_img)
        noise = torch.randn_like(clean_target) * self.noise_std 
        noisy_img = torch.clamp(clean_target + noise, 0., 1.)

        return noisy_img, clean_target

    @staticmethod 
    def load_images(data_dir):
        images = []
        for filename in os.listdir(data_dir):
            try:
                img = Image.open(os.path.join(data_dir, filename)).convert('RGB') # Grayscale
                img_np = np.array(img, dtype=np.float32) / 255.0
                # Convert to [C, H, W] tensor format
                img_torch = torch.from_numpy(img_np).permute(2, 0, 1)  # Change to [C, H, W]
                images.append(img_torch)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        return images

class NoisyCIFAR10(Dataset):
    def __init__(self, root='./data', train=True, noise_std=0.3):
        super().__init__()
        self.noise_std = noise_std
        self.transform = transforms.ToTensor()
        self.cifar_dataset = datasets.CIFAR10(
            root=root,
            train=train,
            download=True
        )

    def __len__(self):
        return len(self.cifar_dataset)

    def __getitem__(self, index):
        clean_pil_img, _ = self.cifar_dataset[index]
        clean_target = self.transform(clean_pil_img)
        noise = torch.randn_like(clean_target) * self.noise_std
        noisy_img = torch.clamp(clean_target + noise, 0., 1.)

        return noisy_img, clean_target

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

# --- DEMONSTRATION OF THE FUNCTION ---
if __name__ == '__main__':
    bsd = NoisyBSD68("./Data/BSD68", 1, 224, 0.3) 
    noisy_img, clean_target = bsd[0]

    visualize_denoising_results(clean_target, noisy_img, noisy_img, "./Data/test/bsd_example.png")
