'''Training & Evaluation Module for Convolutional Neural Networks'''

import torch
import torch.nn as nn
import torch.optim as optim

import os
import time 

from utils import set_seed, visualize_denoising_results, visualize_denoising_results_1D

from dataset import MNIST1D_Plot_Extended


def Train_Eval(args, 
               model: nn.Module, 
                dataset
               ):
    
    if args.seed != 0:
        set_seed(args.seed)
    
    criterion = nn.MSELoss()

    
    # Optimizer 
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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

    epoch_results = [] 

        
    # ==================== ROBUST GFLOPs Calculation with PyTorch Profiler ====================
    try:
        import torch.profiler

        # Get a single batch from the train_loader to determine input size
        input_tensor, _ = next(iter(dataset.train_loader))
        input_tensor = input_tensor.to(device)

        # Profile a single forward pass
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            with_flops=True
        ) as prof:
            with torch.no_grad():
                model(input_tensor[0:1])

        # A more robust way to get total FLOPs: sum them up from all events
        total_flops = sum(event.flops for event in prof.key_averages())

        if total_flops > 0:
            gflops = total_flops / 1e9
            params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            params_m = params / 1e6
            print(f"   - Trainable Parameters: {params_m:.8f} M")

            print(f"Model Complexity (Profiler):")
            print(f"   - GFLOPs: {gflops:.8f}")
            print(f"   - Trainable Parameters: {params_m:.8f} M")
            epoch_results.append(f"Model Complexity (Profiler): GFLOPs: {gflops:.8f}, Trainable Parameters: {params_m:.8f} M")
            
        else:
            # If this still fails, fvcore is the best alternative
            print("Profiler returned 0 FLOPs. Consider using the 'fvcore' method instead for a theoretical count.")

    except Exception as e:
        print(f"Could not calculate GFLOPs with PyTorch Profiler: {e}")
    # =====================================================================
    

    # Training Loop
    epoch_times = [] # Average Epoch Time 
    
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
        for noisy_img, img  in dataset.train_loader: 
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

        train_psnr /= len(dataset.train_loader)
        end_time = time.time()
        epoch_result += f"[Epoch {epoch+1:03d}] Time: {end_time - start_time:.4f}s | [Train] Loss: {train_running_loss/len(dataset.train_loader):.8f} PSNR: {train_psnr:.4f}dB | "
        epoch_times.append(end_time - start_time)
        
        # Model Evaluation 
        model.eval()
        test_psnr = 0.0
        
        with torch.no_grad():
            if args.dataset == "cifar10":
                for noisy_img, img in dataset.test_loader: 
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
            else: # CBSD68/BSD68 Single Image Test
                noisy_img, img = dataset.test_data
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
        if args.dataset == "cifar10":
            visualize_denoising_results(img[0], noisy_img[0], outputs[0], save_path)
        

            test_psnr /= len(dataset.test_loader)
            epoch_result += f"[Test] Loss: {test_running_loss/len(dataset.test_loader):.8f} PSNR: {test_psnr:.4f}dB"
            print(epoch_result)
            epoch_results.append(epoch_result)
        else: 
            visualize_denoising_results(img.squeeze(0), noisy_img.squeeze(0), outputs.squeeze(0), save_path)
            epoch_result += f"[Test] Loss: {test_running_loss:.8f} PSNR: {test_psnr:.4f}dB"
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
        
def Train_Eval_1D(args, 
               model: nn.Module, 
               noisy_data, 
               clean_data
               ):
    """Train and evaluate a model for denoising MNIST1D data."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import time
    import os
    from utils import set_seed, measure_psnr
    from dataset import MNIST1D_Plot_Extended
    
    if args.seed != 0:
        set_seed(args.seed)
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Optimizer 
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Learning Rate Scheduler
    scheduler = None
    if args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.total_steps)
    elif args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)
        
    # Device
    device = args.device
    model.to(device)
    criterion.to(device)
    
    # Mixed precision setup
    if args.use_amp:
        scaler = torch.amp.GradScaler()

    # Initialize results list
    epoch_results = []
    epoch_times = []
    
    # Using the provided data setup
    x_train = noisy_data['x'].to(device)
    y_train = clean_data['x'].to(device)

    x_test = noisy_data['x_test'].to(device)
    y_test = clean_data['x_test'].to(device)

    y_labels = clean_data['y_test']
    
    # Tracking metrics
    max_psnr = -float('inf')  # Start with negative infinity
    max_step = 0
    
    # Initialize plotting
    plot = MNIST1D_Plot_Extended()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Training loop
    print(f"Starting training for {args.total_steps} steps with batch size {args.batch_size}...")
    
    for step in range(args.total_steps + 1):
        start_time = time.time()
        
        # Training step
        model.train()
        
        # Get batch (ensure we don't go out of bounds)
        bix = (step * args.batch_size) % (len(x_train) - args.batch_size)
        x_batch = x_train[bix:bix + args.batch_size]  # Noisy inputs
        y_batch = y_train[bix:bix + args.batch_size]  # Clean targets
        
        # Reset gradients
        optimizer.zero_grad()
        
        if args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            if hasattr(args, 'clip_grad_norm') and args.clip_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            
            if hasattr(args, 'clip_grad_norm') and args.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                
            optimizer.step()
        
        # Update learning rate if using scheduler
        if scheduler and not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()
            
        # Log progress periodically
        if step % 1000 == 0:
            end_time = time.time()
            epoch_times.append(end_time - start_time)
            
            # Compute training PSNR
            with torch.no_grad():
                train_psnr = measure_psnr(outputs, y_batch).item()
            
            # CRITICAL FIX: Create a fresh evaluation for test set
            model.eval()
            test_loss = 0.0
            test_psnr = 0.0
            test_count = 0
            
            # Explicit debug print to verify evaluation is happening
            print(f"[DEBUG] Evaluating model at step {step}...")
            
            # Use multiple small batches for evaluation
            with torch.no_grad():
                for i in range(5):  # Use 5 different batches for evaluation
                    test_bix = (i * args.batch_size) % (len(x_test) - args.batch_size)
                    test_x_batch = x_test[test_bix:test_bix + args.batch_size]
                    test_y_batch = y_test[test_bix:test_bix + args.batch_size]
                    
                    # Run model on test data - with explicit new computation
                    test_outputs = model(test_x_batch)
                    
                    # Calculate batch metrics
                    batch_test_loss = criterion(test_outputs, test_y_batch).item()
                    batch_test_psnr = measure_psnr(test_outputs, test_y_batch).item()
                    
                    # Debug print individual batch metrics
                    print(f"[DEBUG] Test batch {i}: Loss={batch_test_loss:.6f}, PSNR={batch_test_psnr:.4f}dB")
                    
                    # Accumulate metrics
                    test_loss += batch_test_loss
                    test_psnr += batch_test_psnr
                    test_count += 1
                
                # Calculate average metrics
                test_loss /= test_count
                test_psnr /= test_count
            
            # Update max PSNR
            if test_psnr > max_psnr:
                max_psnr = test_psnr
                max_step = step
                
                # Save best model
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'test_psnr': test_psnr,
                }, os.path.join(args.output_dir, 'best_model.pt'))
            
            # Log results
            result_str = f"[Step {step:05d}/{args.total_steps}] "
            result_str += f"Time: {end_time - start_time:.4f}s | "
            result_str += f"[Train] Loss: {loss.item():.8f} PSNR: {train_psnr:.4f}dB | "
            result_str += f"[Test] Loss: {test_loss:.8f} PSNR: {test_psnr:.4f}dB"
            print(result_str)
            epoch_results.append(result_str)
            
            # Update scheduler if using plateau
            if scheduler and isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(test_psnr)
            
            # Create visualization using first test batch
            viz_bix = 0
            viz_x_batch = x_test[viz_bix:viz_bix + args.batch_size]
            viz_y_batch = y_test[viz_bix:viz_bix + args.batch_size]
            
            with torch.no_grad():
                viz_outputs = model(viz_x_batch)
            
            save_path = os.path.join(args.output_dir, f"step_{step:05d}.png")
            plot.plot_denoising_results(
                viz_y_batch[:10],        # Clean signals
                viz_x_batch[:10],        # Noisy signals 
                viz_outputs[:10],        # Denoised signals
                y_labels[viz_bix:viz_bix + 10],  # Labels
                clean_data['t'],          # Time values
                save_path,                # Save path
                zoom=5,                   # Zoom level
                title=f"Step {step} Test Results"  # Title
            )
    
    # Final summary
    epoch_results.append("\n--- Training Summary ---")
    if epoch_times:
        epoch_results.append(f"Average Step Time: {sum(epoch_times) / len(epoch_times):.4f}s")
    epoch_results.append(f"Max PSNR: {max_psnr:.4f}dB at Step {max_step}")
    
    print(f"\nTraining complete! Max PSNR: {max_psnr:.4f}dB at Step {max_step}")
    
    return epoch_results

def measure_psnr(img, img2):
    """Computes the PSNR (Peak Signal-to-Noise Ratio) between two images."""
    mse = nn.MSELoss()(img, img2)
    if mse == 0:
        return float('inf')  # If no noise is present, PSNR is infinite
    max_pixel = 1.0  # Assuming the images are normalized between 0 and 1
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr