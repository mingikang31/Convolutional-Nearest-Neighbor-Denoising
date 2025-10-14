#! /bin/bash 
#SBATCH --nodes=1 
#SBATCH --mem=64G
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=denoising-exp-pt-1
#SBATCH --time=500:00:00
#SBATCH --output=slurm_out/%j.out
#SBATCH --error=slurm_out/%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=mkang2@bowdoin.edu

cd /mnt/research/j.farias/mkang2/Convolutional-Nearest-Neighbor-Denoising

source activate mingi

# CBSD68 UNet 

python main.py --model unet --layer Conv2d --channels 32 --num_pool_layers 4 --kernel_size 3 --padding 1 --dataset cbsd68 --data_path ./Data/CBSD68 --noise_std 0.1 --num_epochs 50 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-2 --weight_decay 5e-4 --output_dir ./Output/Oct13_Denoising/CBSD/Unet_Conv2d_K3_S42

python main.py --model unet --layer ConvNN --channels 32 --num_pool_layers 4 --K 9 --sampling_type all --num_samples -1 --shuffle_pattern NA --shuffle_scale 0 --magnitude_type cosine --similarity_type Col --aggregation_type Col --padding 1 --dataset cbsd68 --data_path ./Data/CBSD68 --noise_std 0.1 --num_epochs 50 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-2 --weight_decay 5e-4 --output_dir ./Output/Oct13_Denoising/CBSD/Unet_ConvNN_K9_S42

python main.py --model unet --layer Branching --channels 32 --num_pool_layers 4 --kernel_size 3 --K 9 --sampling_type all --num_samples -1 --shuffle_pattern NA --shuffle_scale 0 --magnitude_type cosine --similarity_type Col --aggregation_type Col --padding 1 --dataset cbsd68 --data_path ./Data/CBSD68 --noise_std 0.1 --num_epochs 50 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-2 --weight_decay 5e-4 --output_dir ./Output/Oct13_Denoising/CBSD/Unet_Branching_K9_S42

# CIFAR10 UNet 

python main.py --model unet --layer Conv2d --channels 32 --num_pool_layers 4 --kernel_size 3 --padding 1 --dataset cifar10  --noise_std 0.1 --num_epochs 50 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-2 --weight_decay 5e-4 --output_dir ./Output/Oct13_Denoising/CIFAR10/Unet_Conv2d_K3_S42

python main.py --model unet --layer ConvNN --channels 32 --num_pool_layers 4 --K 9 --sampling_type all --num_samples -1 --shuffle_pattern NA --shuffle_scale 0 --magnitude_type cosine --similarity_type Col --aggregation_type Col --padding 1 --dataset cifar10 --noise_std 0.1 --num_epochs 50 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-2 --weight_decay 5e-4 --output_dir ./Output/Oct13_Denoising/CIFAR10/Unet_ConvNN_K9_S42

python main.py --model unet --layer Branching --channels 32 --num_pool_layers 4 --kernel_size 3 --K 9 --sampling_type all --num_samples -1 --shuffle_pattern NA --shuffle_scale 0 --magnitude_type cosine --similarity_type Col --aggregation_type Col --padding 1 --dataset cifar10 --noise_std 0.1 --num_epochs 50 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-2 --weight_decay 5e-4 --output_dir ./Output/Oct13_Denoising/CIFAR10/Unet_Branching_K9_S42


# CBSD68 DnCNN 

python main.py --model dncnn --layer Conv2d --channels 32 --num_layers 17 --kernel_size 3 --padding 1 --dataset cbsd68 --data_path ./Data/CBSD68 --noise_std 0.1 --num_epochs 50 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-2 --weight_decay 5e-4 --output_dir ./Output/Oct13_Denoising/CBSD/DnCNN_Conv2d_K3_S42

python main.py --model dncnn --layer ConvNN --channels 32 --num_layers 17 --K 9 --sampling_type all --num_samples -1 --shuffle_pattern NA --shuffle_scale 0 --magnitude_type cosine --similarity_type Col --aggregation_type Col --padding 1 --dataset cbsd68 --data_path ./Data/CBSD68 --noise_std 0.1 --num_epochs 50 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-2 --weight_decay 5e-4 --output_dir ./Output/Oct13_Denoising/CBSD/DnCNN_ConvNN_K9_S42

python main.py --model dncnn --layer Branching --channels 32 --num_layers 17 --kernel_size 3 --K 9 --sampling_type all --num_samples -1 --shuffle_pattern NA --shuffle_scale 0 --magnitude_type cosine --similarity_type Col --aggregation_type Col --padding 1 --dataset cbsd68 --data_path ./Data/CBSD68 --noise_std 0.1 --num_epochs 50 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-2 --weight_decay 5e-4 --output_dir ./Output/Oct13_Denoising/CBSD/DnCNN_Branching_K9_S42

# CIFAR10 DnCNN

python main.py --model dncnn --layer Conv2d --channels 32 --num_layers 17 --kernel_size 3 --padding 1 --dataset cifar10 --noise_std 0.1 --num_epochs 50 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-2 --weight_decay 5e-4 --output_dir ./Output/Oct13_Denoising/CIFAR10/DnCNN_Conv2d_K3_S42

python main.py --model dncnn --layer ConvNN --channels 32 --num_layers 17 --K 9 --sampling_type all --num_samples -1 --shuffle_pattern NA --shuffle_scale 0 --magnitude_type cosine --similarity_type Col --aggregation_type Col --padding 1 --dataset cifar10 --noise_std 0.1 --num_epochs 50 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-2 --weight_decay 5e-4 --output_dir ./Output/Oct13_Denoising/CIFAR10/DnCNN_ConvNN_K9_S42

python main.py --model dncnn --layer Branching --channels 32 --num_layers 17 --kernel_size 3 --K 9 --sampling_type all --num_samples -1 --shuffle_pattern NA --shuffle_scale 0 --magnitude_type cosine --similarity_type Col --aggregation_type Col --padding 1 --dataset cifar10 --noise_std 0.1 --num_epochs 50 --seed 42 --padding 1 --lr_step 2 --lr_gamma 0.95 --lr 1e-2 --weight_decay 5e-4 --output_dir ./Output/Oct13_Denoising/CIFAR10/DnCNN_Branching_K9_S42
