"""Main File for the project"""

import argparse 
from pathlib import Path
import os 

# Datasets 
from dataset import Denoise_CIFAR10, Denoise_BSD68, Denoise_CBSD68
from train_eval import Train_Eval

# Models 
from models.allconvnet import AllConvNet 

# Utilities 
from utils import write_to_file, set_seed


def args_parser():
    parser = argparse.ArgumentParser(description="Convolutional Nearest Neighbor Denoising training and evaluation", add_help=False) 
    
    # Model Arguments
    parser.add_argument("--layer", type=str, default="ConvNN", choices=["Conv2d", "Conv2d_New", "Conv2d_New_1d", "ConvNN", "ConvNN_Attn", "Attention", "Conv2d/ConvNN", "Conv2d/ConvNN_Attn", "Attention/ConvNN", "Attention/ConvNN_Attn", "Conv2d/Attention"], help="Type of Convolution or Attention layer to use")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of layers.")   
    parser.add_argument("--channels", nargs='+', type=int, default=[8, 16, 32], help="Channel sizes for each layer.")
    
    # ConvNN Layer Arguments
    parser.add_argument("--K", type=int, default=9, help="K-nearest neighbor for ConvNN")
    parser.add_argument("--kernel_size", type=int, default=3, help="Kernel Size for Conv2d")        
    parser.add_argument("--sampling_type", type=str, default='all', choices=["all", "random", "spatial"], help="Sampling method for ConvNN Models")
    parser.add_argument("--num_samples", type=int, default=-1, help="Number of samples for ConvNN Models")
    parser.add_argument("--sample_padding", type=int, default=0, help="Padding for spatial sampling in ConvNN Models")
    parser.add_argument("--shuffle_pattern", type=str, default="NA", choices=["BA", "NA"], help="Shuffle pattern: BA (Before & After) or NA (No Shuffle)")
    parser.add_argument("--shuffle_scale", type=int, default=2, help="Shuffle scale for ConvNN Models")
    parser.add_argument("--magnitude_type", type=str, default="similarity", choices=["similarity", "distance"], help="Magnitude type for ConvNN Models")
    parser.add_argument("--coordinate_encoding", action="store_true", help="Use coordinate encoding in ConvNN Models")
    parser.set_defaults(coordinate_encoding=False)
    
    # ConvNN Attention Arguments
    parser.add_argument("--num_heads", type=int, default=4, help="Number of heads for Attention Models")    
    parser.add_argument("--attention_dropout", type=float, default=0.1, help="Dropout rate for the model")    
                                                                                                                                    
    # Arguments for Data 
    parser.add_argument("--dataset", type=str, default="bsd68", choices=["cifar10", "bsd68", 'cbsd68', 'mnist1d'], help="Dataset to use for training and evaluation")
    parser.add_argument("--data_path", type=str, default="./Data/BSD68", help="Path to the dataset")
    parser.add_argument("--noise_std", type=float, default=0.2, help="Standard deviation of Gaussian noise")
    
    # Training Arguments
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training and evaluation")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs for training")
    parser.add_argument("--use_amp", action="store_true", help="Use mixed precision training")
    parser.set_defaults(use_amp=False)
    parser.add_argument("--clip_grad_norm", type=float, default=None, help="Gradient clipping value")
        
    # Optimizer Arguments 
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='Default Optimizer: adam')
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
    
    return parser
    
def main(args):
    
    # Check if the output directory exists, if not create it
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Dataset 
    if args.dataset == "cifar10":
        dataset = Denoise_CIFAR10(args.data_path, args.batch_size, args.noise_std)
        args.img_size = dataset.shape()
    elif args.dataset == "bsd68":
        dataset = Denoise_BSD68(args.data_path, args.batch_size, args.noise_std, 4000, 800, 128)
        args.img_size = dataset.shape()
    elif args.dataset == "cbsd68":
        dataset = Denoise_CBSD68(args.data_path, args.batch_size, args.noise_std, 4000, 800, 128)
        args.img_size = dataset.shape()
    elif args.dataset == "mnist1d":
        pass

        
    # Model 
    model = AllConvNet(args)

    print(f"Model: {model.name}")

    # Parameters
    total_params, trainable_params = model.parameter_count()
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")
    args.total_params = total_params
    args.trainable_params = trainable_params
    
    # Set the seed for reproducibility
    set_seed(args.seed)
    
    
    # Training Modules 
    train_eval_results = Train_Eval(args, 
                                model, 
                                dataset.train_loader, 
                                dataset.test_loader
                                )
    
    # Storing Results in output directory 
    write_to_file(os.path.join(args.output_dir, "args.txt"), args)
    write_to_file(os.path.join(args.output_dir, "model.txt"), model)
    write_to_file(os.path.join(args.output_dir, "train_eval_results.txt"), train_eval_results)

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description="Convolutional Nearest Neighbor training and evaluation", parents=[args_parser()])
    args = parser.parse_args()
    main(args)

