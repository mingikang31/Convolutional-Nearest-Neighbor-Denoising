'''Training & Evaluation Module for Convolutional Neural Networks'''

import torch
import torch.nn as nn
import torch.optim as optim

import os
from tqdm import tqdm
import time 

from utils import set_seed, visualize_denoising_results



def Train_Eval(args, 
               model: nn.Module, 
               train_loader, 
               test_loader
               ):
    
    if args.seed != 0:
        set_seed(args.seed)
    
    criterion = nn.MSELoss()

    
    # Optimizer 
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Learning Rate Scheduler
    scheduler = None
    if args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    elif args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)

        
    # Device
    device = args.device
    model.to(device)
    criterion.to(device)
    
    if args.use_amp:
        scaler = torch.amp.GradScaler("cuda")

    # Training Loop
    epoch_times = [] # Average Epoch Time 
    epoch_results = [] 
    
    max_psnr = 0.0 
    max_epoch = 0
    
    for epoch in range(args.num_epochs):
        # Model Training
        model.train() 
        train_running_loss = 0.0
        test_running_loss = 0.0
        epoch_result = ""
        
        start_time = time.time()

        train_psnr = 0.0
        for noisy_img, img  in train_loader: 
            img, noisy_img = img.to(device), noisy_img.to(device)
            optimizer.zero_grad()
            
            # use mixed precision training
            if args.use_amp:
                with torch.amp.autocast('cuda'):
                    output_img = model(noisy_img)
                    loss = criterion(output_img, img)
                scaler.scale(loss).backward()
                if hasattr(args, 'clip_grad_norm') and args.clip_grad_norm is not None:
                    scaler.unscale_(optimizer) # Unscale gradients before clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:    
                output_img = model(noisy_img)
                loss = criterion(output_img, img)
                loss.backward()
                if hasattr(args, 'clip_grad_norm') and args.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()            

            psnr = measure_psnr(output_img, img)
            train_psnr += psnr.item()
            train_running_loss += loss.item()

        train_psnr /= len(train_loader)
        end_time = time.time()
        epoch_result += f"[Epoch {epoch+1:03d}] Time: {end_time - start_time:.4f}s | [Train] Loss: {train_running_loss/len(train_loader):.8f} PSNR: {train_psnr:.4f}dB | "
        epoch_times.append(end_time - start_time)
        
        # Model Evaluation 
        model.eval()
        test_psnr = 0.0
        with torch.no_grad():
            for noisy_img, img in test_loader: 
                img, noisy_img = img.to(device), noisy_img.to(device)
                if args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(noisy_img)
                else: 
                    outputs = model(noisy_img)
                loss = criterion(outputs, img)
                test_running_loss += loss.item()

                psnr = measure_psnr(outputs, img)
                test_psnr += psnr.item()

        # Save Visual
        save_path = os.path.join(args.output_dir, f"test_epoch_{epoch+1}.png")
        visualize_denoising_results(img[0], noisy_img[0], outputs[0], save_path)

        test_psnr /= len(test_loader)
        epoch_result += f"[Test] Loss: {test_running_loss/len(test_loader):.8f} PSNR: {test_psnr:.4f}dB"
        print(epoch_result)
        epoch_results.append(epoch_result)

        # Max PSNR Check
        if test_psnr > max_psnr:
            max_psnr = test_psnr
            max_epoch = epoch + 1
            
        # Learning Rate Scheduler Step
        if scheduler: 
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(test_psnr)
            else:
                scheduler.step()
                
                
    epoch_results.append(f"\nAverage Epoch Time: {sum(epoch_times) / len(epoch_times):.4f}s")
    epoch_results.append(f"Max PSNR: {max_psnr:.4f}dB at Epoch {max_epoch}")
    
    return epoch_results
        

def measure_psnr(img, img2):
    """Computes the PSNR (Peak Signal-to-Noise Ratio) between two images."""
    mse = nn.MSELoss()(img, img2)
    if mse == 0:
        return float('inf')  # If no noise is present, PSNR is infinite
    max_pixel = 1.0  # Assuming the images are normalized between 0 and 1
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr
