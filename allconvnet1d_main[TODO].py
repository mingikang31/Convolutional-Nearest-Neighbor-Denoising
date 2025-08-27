"""Main File for the project"""

import argparse 
from pathlib import Path
import os 

# Datasets 
from dataset import MNIST1D
from train_eval import Train_Eval

# Models 
from models.allconvnet1d import AllConvNet1D

# Utilities 
from utils import write_to_file, set_seed

"""
Only doing Conv2d, Conv2d_New, Conv2d_New_1d, ConvNN, and ConvNN_Attn
"""

def args_parser():
    parser = argparse.ArgumentParser(description="Convolutional Nearest Neighbor training and evaluation", add_help=False) 
    
    # Model Arguments
    parser.add_argument("--layer", type=str, default="Conv1d", choices=["Conv1d", "Conv1d_New", "ConvNN", "ConvNN_Attn", "Attention", "Conv2d/ConvNN", "Conv2d/ConvNN_Attn", "Attention/ConvNN", "Attention/ConvNN_Attn", "Conv2d/Attention"], help="Type of Convolution or Attention layer to use")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of layers.")   
    parser.add_argument("--channels", nargs='+', type=int, default=[32, 16, 8], help="Channel sizes for each layer.")

    # Additional Layer Arguments
    parser.add_argument("--K", type=int, default=3, help="K-nearest neighbor for ConvNN")
    parser.add_argument("--kernel_size", type=int, default=3, help="Kernel Size for Conv2d")        
    parser.add_argument("--sampling_type", type=str, default='all', choices=["all", "random", "spatial"], help="Sampling method for ConvNN Models")
    parser.add_argument("--num_samples", type=int, default=-1, help="Number of samples for ConvNN Models")
    parser.add_argument("--sample_padding", type=int, default=0, help="Padding for spatial sampling in ConvNN Models")
    
    parser.add_argument("--num_heads", type=int, default=4, help="Number of heads for Attention Models")    
    parser.add_argument("--attention_dropout", type=float, default=0.1, help="Dropout rate for the model")    

    parser.add_argument("--shuffle_pattern", type=str, default="NA", choices=["BA", "NA"], help="Shuffle pattern: BA (Before & After) or NA (No Shuffle)")
    parser.add_argument("--shuffle_scale", type=int, default=2, help="Shuffle scale for ConvNN Models")
    parser.add_argument("--magnitude_type", type=str, default="similarity", choices=["similarity", "distance"], help="Magnitude type for ConvNN Models")
    parser.add_argument("--coordinate_encoding", action="store_true", help="Use coordinate encoding in ConvNN Models")
    parser.set_defaults(coordinate_encoding=False)

    # Arguments for Data 
            
    # Training Arguments
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training and evaluation")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs for training")
    parser.add_argument("--use_amp", action="store_true", help="Use mixed precision training")
    parser.set_defaults(use_amp=False)
    parser.add_argument("--clip_grad_norm", type=float, default=None, help="Gradient clipping value")
    
    
    # Loss Function Arguments
    parser.add_argument("--criterion", type=str, default="CrossEntropy", choices=["CrossEntropy", "MSE"], help="Loss function to use for training")
    
    # Optimizer Arguments 
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'sgd', 'adamw'], help='Default Optimizer: adamw')
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
    parser.add_argument("--output_dir", type=str, default="./Output/MNIST1D/AllConvNet1D/ConvNN", help="Directory to save the output files")

    return parser

def check_args(args):
    # Check the arguments based on the model 
    print("Checking arguments based on the model...")    
    
    assert args.num_layers == len(args.channels), f"Number of layers {args.num_layers} does not match the number of channels {len(args.channels)}"
        
    if args.sampling_type == "all": # only for Conv2d_NN, Conv2d_NN_Attn
        args.num_samples = -1
    if args.num_samples == -1:
        args.sampling_type = "all"

    args.resize = False
    return args
    
    
def main(args):

    args = check_args(args)
    
    # Check if the output directory exists, if not create it
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Dataset 
    dataset = MNIST1D(args)
    args.num_classes = dataset.num_classes
    args.img_size = dataset.img_size

    
    # Model 
    model = AllConvNet1D(args)

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

# python allconvnet_main.py --layer Conv2d --num_layers 3 --channels 8 16 32 --dataset cifar10 --num_epochs 10 --device mps --output_dir ./Output/Simple/Conv2d 

# python allconvnet_main.py --layer Conv2d/ConvNN --num_layers 3 --channels 8 16 32 --sampling Spatial --num_samples 8 --dataset cifar10 --num_epochs 10 --device cuda --output_dir ./Output/Simple/Conv2d_ConvNN_Spatial
