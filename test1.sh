#! /bin/bash 
#SBATCH --nodes=1 
#SBATCH --mem=64G
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=default-exps
#SBATCH --time=500:00:00
#SBATCH --output=slurm_out/%j.out
#SBATCH --error=slurm_out/%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=mkang2@bowdoin.edu

cd /mnt/research/j.farias/mkang2/Convolutional-Nearest-Neighbor-Denoising

source activate mingi

# # All Convolutional Network
# BSD68
python allconvnet_main.py --layer Conv2d --num_layers 4 --channels 64 32 16 8 --dataset bsd68 --data_path ./Data/BSD68 --noise_std 0.2 --batch_size 64 --num_epochs 100 --lr 1e-3 --output_dir ./Output/Aug_25_31/BSD68/ACM/Conv2d

python allconvnet_main.py --layer ConvNN --sampling_type all --K 9 --num_layers 4 --channels 64 32 16 8 --dataset bsd68 --data_path ./Data/BSD68 --noise_std 0.2 --batch_size 64 --num_epochs 100 --lr 1e-3 --output_dir ./Output/Aug_25_31/BSD68/ACM/ConvNN_All_K9

python allconvnet_main.py --layer ConvNN --sampling_type random --K 9 --num_samples 64 --num_layers 4 --channels 64 32 16 8 --dataset bsd68 --data_path ./Data/BSD68 --noise_std 0.2 --batch_size 64 --num_epochs 100 --lr 1e-3 --output_dir ./Output/Aug_25_31/BSD68/ACM/ConvNN_Random_K9_N64 

python allconvnet_main.py --layer ConvNN --sampling_type spatial --K 9 --num_samples 8 --num_layers 4 --channels 64 32 16 8 --dataset bsd68 --data_path ./Data/BSD68 --noise_std 0.2 --batch_size 64 --num_epochs 100 --lr 1e-3 --output_dir ./Output/Aug_25_31/BSD68/ACM/ConvNN_Spatial_K9_N8


# CBSD68
python allconvnet_main.py --layer Conv2d --num_layers 4 --channels 64 32 16 8 --dataset cbsd68 --data_path ./Data/CBSD68 --noise_std 0.2 --batch_size 64 --num_epochs 100 --lr 1e-3 --output_dir ./Output/Aug_25_31/CBSD68/ACM/Conv2d 

python allconvnet_main.py --layer ConvNN --sampling_type all --K 9 --num_layers 4 --channels 64 32 16 8 --dataset cbsd68 --data_path ./Data/CBSD68 --noise_std 0.2 --batch_size 64 --num_epochs 100 --lr 1e-3 --output_dir ./Output/Aug_25_31/CBSD68/ACM/ConvNN_All_K9

python allconvnet_main.py --layer ConvNN --sampling_type random --K 9 --num_samples 64 --num_layers 4 --channels 64 32 16 8 --dataset cbsd68 --data_path ./Data/CBSD68 --noise_std 0.2 --batch_size 64 --num_epochs 100 --lr 1e-3 --output_dir ./Output/Aug_25_31/CBSD68/ACM/ConvNN_Random_K9_N64

python allconvnet_main.py --layer ConvNN --sampling_type spatial --K 9 --num_samples 8 --num_layers 4 --channels 64 32 16 8 --dataset cbsd68 --data_path ./Data/CBSD68 --noise_std 0.2 --batch_size 64 --num_epochs 100 --lr 1e-3 --output_dir ./Output/Aug_25_31/CBSD68/ACM/ConvNN_Spatial_K9_N8

# # UNet
# BSD68
python unet_main.py --layer Conv2d --chans 32 --num_pool_layers 4 --drop_prob 0.1 --dataset bsd68 --data_path ./Data/BSD68 --noise_std 0.2 --batch_size 64 --num_epochs 100 --lr 1e-3 --output_dir ./Output/Aug_25_31/BSD68/UNet/Conv2d 

python unet_main.py --layer ConvNN --sampling_type all --K 9 --chans 32 --num_pool_layers 4 --drop_prob 0.1 --dataset bsd68 --data_path ./Data/BSD68 --noise_std 0.2 --batch_size 64 --num_epochs 100 --lr 1e-3 --output_dir ./Output/Aug_25_31/BSD68/UNet/ConvNN_All_K9

python unet_main.py --layer ConvNN --sampling_type random --K 9 --num_samples 64 --chans 32 --num_pool_layers 4 --drop_prob 0.1 --dataset bsd68 --data_path ./Data/BSD68 --noise_std 0.2 --batch_size 64 --num_epochs 100 --lr 1e-3 --output_dir ./Output/Aug_25_31/BSD68/UNet/ConvNN_Random_K9_N64

python unet_main.py --layer ConvNN --sampling_type spatial --K 9 --num_samples 8 --chans 32 --num_pool_layers 4 --drop_prob 0.1 --dataset bsd68 --data_path ./Data/BSD68 --noise_std 0.2 --batch_size 64 --num_epochs 100 --lr 1e-3 --output_dir ./Output/Aug_25_31/BSD68/UNet/ConvNN_Spatial_K9_N8

# CBSD68
python unet_main.py --layer Conv2d --chans 32 --num_pool_layers 4 --drop_prob 0.1 --dataset cbsd68 --data_path ./Data/CBSD68 --noise_std 0.2 --batch_size 64 --num_epochs 100 --lr 1e-3 --output_dir ./Output/Aug_25_31/CBSD68/UNet/Conv2d

python unet_main.py --layer ConvNN --sampling_type all --K 9 --chans 32 --num_pool_layers 4 --drop_prob 0.1 --dataset cbsd68 --data_path ./Data/CBSD68 --noise_std 0.2 --batch_size 64 --num_epochs 100 --lr 1e-3 --output_dir ./Output/Aug_25_31/CBSD68/UNet/ConvNN_All_K9

python unet_main.py --layer ConvNN --sampling_type random --K 9 --num_samples 64 --chans 32 --num_pool_layers 4 --drop_prob 0.1 --dataset cbsd68 --data_path ./Data/CBSD68 --noise_std 0.2 --batch_size 64 --num_epochs 100 --lr 1e-3 --output_dir ./Output/Aug_25_31/CBSD68/UNet/ConvNN_Random_K9_N64

python unet_main.py --layer ConvNN --sampling_type spatial --K 9 --num_samples 8 --chans 32 --num_pool_layers 4 --drop_prob 0.1 --dataset cbsd68 --data_path ./Data/CBSD68 --noise_std 0.2 --batch_size 64 --num_epochs 100 --lr 1e-3 --output_dir ./Output/Aug_25_31/CBSD68/UNet/ConvNN_Spatial_K9_N8


