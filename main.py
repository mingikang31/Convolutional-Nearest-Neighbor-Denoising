"""Main File for the project"""

import argparse 
from pathlib import Path
import os

import torch 

# Datasets 
from dataset import Denoise_CIFAR10, Denoise_BSD68, Denoise_CBSD68
from train_eval import Train_Eval

# Models 
from models.unet import UNet
from models.dncnn import DnCNN

# Utilities 
from utils import write_to_file, set_seed


def args_parser():
    parser = argparse.ArgumentParser(description="Convolutional Nearest Neighbor Denoising training and evaluation", add_help=False) 
    
    # Model Arguments
    parser.add_argument("--model", type=str, default="unet", choices=["unet", "dncnn"], help="Model to use for training and evaluation")
    
    parser.add_argument("--layer", type=str, default="ConvNN", choices=["Conv2d", "ConvNN", "Branching"], help="Type of Convolution layer to use")
    parser.add_argument("--channels", type=int, default=32, help="Number of channels in the U-Net")

    # UNet Specific Arguments
    parser.add_argument("--num_pool_layers", type=int, default=4, help="Number of pooling layers in the U-Net") ## CIFAR Max = 3, other = 4

    # DnCNN Specific Arguments
    parser.add_argument("--num_layers", type=int, default=17, help="Number of layers in the DnCNN")

    # ConvNN Layer Arguments
    parser.add_argument("--K", type=int, default=9, help="K-nearest neighbor for ConvNN")
    parser.add_argument("--kernel_size", type=int, default=3, help="Kernel Size for Conv2d")        
    parser.add_argument("--padding", type=int, default=1, help="Padding for ConvNN")
    parser.add_argument("--sampling_type", type=str, default='all', choices=["all", "random", "spatial"], help="Sampling method for ConvNN Models")
    parser.add_argument("--num_samples", type=int, default=-1, help="Number of samples for ConvNN Models")
    parser.add_argument("--sample_padding", type=int, default=0, help="Padding for spatial sampling in ConvNN Models")
    
    # ConvNN specific arguments
    parser.add_argument("--shuffle_pattern", type=str, default="NA", choices=["BA", "NA"], help="Shuffle pattern: BA (Before & After) or NA (No Shuffle)")
    parser.add_argument("--shuffle_scale", type=int, default=0, help="Shuffle scale for ConvNN Models")
    parser.add_argument("--magnitude_type", type=str, default="cosine", choices=["cosine", "euclidean"], help="Magnitude type for ConvNN Models")
    parser.add_argument("--similarity_type", type=str, default="Col", choices=["Loc", "Col", "Loc_Col"], help="Similarity type for ConvNN Models")
    parser.add_argument("--aggregation_type", type=str, default="Col", choices=["Col", "Loc_Col"], help="Aggregation type for ConvNN Models")
    parser.add_argument("--lambda_param", type=float, default=0.5, help="Lambda parameter for Loc_Col aggregation in ConvNN Models")
    parser.add_argument("--branch_ratio", type=float, default=0.5, help="Branch ratio for Branching layer (between 0 and 1), ex. 0.25 means 25% of in_channels and out_channels go to ConvNN branch, rest to Conv2d branch")                                                                                                          
    # Arguments for Data 
    parser.add_argument("--dataset", type=str, default="bsd68", choices=["cifar10", "bsd68", 'cbsd68', 'mnist1d'], help="Dataset to use for training and evaluation")
    parser.add_argument("--data_path", type=str, default="./Data/BSD68", help="Path to the dataset")
    parser.add_argument("--noise_std", type=float, default=0.2, help="Standard deviation of Gaussian noise")
    parser.add_argument("--img_size", type=int, default=64, help="Image size for training and evaluation")
    
    # Training Arguments
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training and evaluation")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs for training")
    parser.add_argument("--use_amp", action="store_true", help="Use mixed precision training")
    parser.set_defaults(use_amp=False)
    parser.add_argument("--clip_grad_norm", type=float, default=None, help="Gradient clipping value")
        
    # Optimizer Arguments 
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'sgd', 'adamw'], help='Default Optimizer: adam')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay for optimizer')
    
    # Learning Rate Arguments
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the optimizer")
    parser.add_argument('--lr_step', type=int, default=20, help='Step size for learning rate scheduler')
    parser.add_argument('--lr_gamma', type=float, default=0.1, help='Gamma for learning rate scheduler')
    parser.add_argument('--scheduler', type=str, default='step', choices=['step', 'cosine', 'plateau'], help='Learning rate scheduler')
    
    # Device Arguments
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"], help="Device to use for training and evaluation")
    parser.add_argument('--seed', default=0, type=int)
    
    # Output Arguments 
    parser.add_argument("--output_dir", type=str, default="./Output/UNet/Conv2d", help="Directory to save the output files")
    
    # Test Arguments
    parser.add_argument("--test_only", action="store_true", help="Only test the model")
    parser.set_defaults(test_only=False)
    return parser
    
def main(args):
    

    # Dataset 
    if args.dataset == "cifar10":
        dataset = Denoise_CIFAR10(args.data_path, args.batch_size, args.noise_std)
        args.img_size = dataset.shape()
    elif args.dataset == "bsd68":
        dataset = Denoise_BSD68(args.data_path, args.batch_size, args.noise_std, 50000, args.img_size)
        args.img_size = dataset.shape()
    elif args.dataset == "cbsd68":
        dataset = Denoise_CBSD68(args.data_path, args.batch_size, args.noise_std, 50000, args.img_size)
        args.img_size = dataset.shape()
    elif args.dataset == "mnist1d":
        pass

        
    # Model 
    if args.model == "dncnn":
        model = DnCNN(args)
    elif args.model == "unet":
        model = UNet(args)

    model = model.to(args.device)
    print(f"Model: {model.name}")

    # Parameters
    total_params, trainable_params = model.parameter_count()
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")
    args.total_params = total_params
    args.trainable_params = trainable_params
    
    if args.test_only:
        ex = torch.Tensor(3, 1, 64, 64) if args.dataset == "bsd68" else torch.Tensor(3, 3, 64, 64).to(args.device)
        out = model(ex)
        print(f"Output shape: {out.shape}")
        print("Testing Complete")
    else:
        # Check if the output directory exists, if not create it
        if args.output_dir:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)

        # Set the seed for reproducibility
        set_seed(args.seed)
        
        
        # Training Modules 
        train_eval_results = Train_Eval(args, 
                                    model, 
                                    dataset, 
                                    )
        
        # Storing Results in output directory 
        write_to_file(os.path.join(args.output_dir, "args.txt"), args)
        write_to_file(os.path.join(args.output_dir, "model.txt"), model)
        write_to_file(os.path.join(args.output_dir, "train_eval_results.txt"), train_eval_results)

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description="Convolutional Nearest Neighbor training and evaluation", parents=[args_parser()])
    args = parser.parse_args()
    main(args)

