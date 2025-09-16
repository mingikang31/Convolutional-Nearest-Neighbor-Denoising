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


# Unet BSD68
python unet_main.py --layer Conv2d --chans 32 --num_pool_layers 4 --drop_prob 0.1 --kernel_size 3 --padding 1 --dataset bsd68 --noise_std 0.05 --num_epochs 50 --lr 1e-4 --lr_step 2 --lr_gamma 0.95 --scheduler step --seed 42 --output_dir ./Output/Sep_15_Unet/BSD/Conv2d_K3_S42

# Col Col 
python unet_main.py --layer ConvNN --chans 32 --num_pool_layers 4 --drop_prob 0.1 --K 9 --padding 1 --sampling_type all --num_samples -1 --shuffle_pattern NA --shuffle_scale 0 --magnitude_type cosine --similarity_type Col --aggregation_type Col --dataset bsd68 --noise_std 0.05 --num_epochs 50 --lr 1e-4 --lr_step 2 --lr_gamma 0.95 --scheduler step --seed 42 --output_dir ./Output/Sep_15_Unet/BSD/ConvNN_Cos_All_K9_Col_Col_S42

python unet_main.py --layer ConvNN --chans 32 --num_pool_layers 4 --drop_prob 0.1 --K 9 --padding 1 --sampling_type all --num_samples -1 --shuffle_pattern NA --shuffle_scale 0 --magnitude_type euclidean --similarity_type Col --aggregation_type Col --dataset bsd68 --noise_std 0.05 --num_epochs 50 --lr 1e-4 --lr_step 2 --lr_gamma 0.95 --scheduler step --seed 42 --output_dir ./Output/Sep_15_Unet/BSD/ConvNN_Euc_All_K9_Col_Col_S42

# Loc Col 
python unet_main.py --layer ConvNN --chans 32 --num_pool_layers 4 --drop_prob 0.1 --K 9 --padding 1 --sampling_type all --num_samples -1 --shuffle_pattern NA --shuffle_scale 0 --magnitude_type cosine --similarity_type Loc --aggregation_type Col --dataset bsd68 --noise_std 0.05 --num_epochs 50 --lr 1e-4 --lr_step 2 --lr_gamma 0.95 --scheduler step --seed 42 --output_dir ./Output/Sep_15_Unet/BSD/ConvNN_Cos_All_K9_Loc_Col_S42

python unet_main.py --layer ConvNN --chans 32 --num_pool_layers 4 --drop_prob 0.1 --K 9 --padding 1 --sampling_type all --num_samples -1 --shuffle_pattern NA --shuffle_scale 0 --magnitude_type euclidean --similarity_type Loc --aggregation_type Col --dataset bsd68 --noise_std 0.05 --num_epochs 50 --lr 1e-4 --lr_step 2 --lr_gamma 0.95 --scheduler step --seed 42 --output_dir ./Output/Sep_15_Unet/BSD/ConvNN_Euc_All_K9_Loc_Col_S42

# Loc_Col Col 
python unet_main.py --layer ConvNN --chans 32 --num_pool_layers 4 --drop_prob 0.1 --K 9 --padding 1 --sampling_type all --num_samples -1 --shuffle_pattern NA --shuffle_scale 0 --magnitude_type cosine --similarity_type Loc_Col --aggregation_type Col --dataset bsd68 --noise_std 0.05 --num_epochs 50 --lr 1e-4 --lr_step 2 --lr_gamma 0.95 --scheduler step --seed 42 --output_dir ./Output/Sep_15_Unet/BSD/ConvNN_Cos_All_K9_LocCol_Col_S42

python unet_main.py --layer ConvNN --chans 32 --num_pool_layers 4 --drop_prob 0.1 --K 9 --padding 1 --sampling_type all --num_samples -1 --shuffle_pattern NA --shuffle_scale 0 --magnitude_type euclidean --similarity_type Loc_Col --aggregation_type Col --dataset bsd68 --noise_std 0.05 --num_epochs 50 --lr 1e-4 --lr_step 2 --lr_gamma 0.95 --scheduler step --seed 42 --output_dir ./Output/Sep_15_Unet/BSD/ConvNN_Euc_All_K9_LocCol_Col_S42

# Loc_Col Loc_Col 
python unet_main.py --layer ConvNN --chans 32 --num_pool_layers 4 --drop_prob 0.1 --K 9 --padding 1 --sampling_type all --num_samples -1 --shuffle_pattern NA --shuffle_scale 0 --magnitude_type cosine --similarity_type Loc_Col --aggregation_type Loc_Col --dataset bsd68 --noise_std 0.05 --num_epochs 50 --lr 1e-4 --lr_step 2 --lr_gamma 0.95 --scheduler step --seed 42 --output_dir ./Output/Sep_15_Unet/BSD/ConvNN_Cos_All_K9_Col_Col_S42

python unet_main.py --layer ConvNN --chans 32 --num_pool_layers 4 --drop_prob 0.1 --K 9 --padding 1 --sampling_type all --num_samples -1 --shuffle_pattern NA --shuffle_scale 0 --magnitude_type euclidean --similarity_type Loc_Col --aggregation_type Loc_Col --dataset bsd68 --noise_std 0.05 --num_epochs 50 --lr 1e-4 --lr_step 2 --lr_gamma 0.95 --scheduler step --seed 42 --output_dir ./Output/Sep_15_Unet/BSD/ConvNN_Euc_All_K9_Loc_Col_Loc_Col_S42


# Unet 
python unet_main.py --layer Conv2d --chans 32 --num_pool_layers 4 --drop_prob 0.1 --kernel_size 3 --padding 1 --dataset bsd68 --noise_std 0.05 --num_epochs 50 --lr 1e-4 --lr_step 2 --lr_gamma 0.95 --scheduler step --seed 42 --output_dir ./Output/Sep_15_Unet/BSD/Conv2d_K3_S42



# cbsd68


# Col Col 
python unet_main.py --layer ConvNN --chans 32 --num_pool_layers 4 --drop_prob 0.1 --K 9 --padding 1 --sampling_type all --num_samples -1 --shuffle_pattern NA --shuffle_scale 0 --magnitude_type cosine --similarity_type Col --aggregation_type Col --dataset cbsd68 --noise_std 0.05 --num_epochs 50 --lr 1e-4 --lr_step 2 --lr_gamma 0.95 --scheduler step --seed 42 --output_dir ./Output/Sep_15_Unet/CBSD/ConvNN_Cos_All_K9_Col_Col_S42 --data_path ./Data/CBSD68

python unet_main.py --layer ConvNN --chans 32 --num_pool_layers 4 --drop_prob 0.1 --K 9 --padding 1 --sampling_type all --num_samples -1 --shuffle_pattern NA --shuffle_scale 0 --magnitude_type euclidean --similarity_type Col --aggregation_type Col --dataset cbsd68 --noise_std 0.05 --num_epochs 50 --lr 1e-4 --lr_step 2 --lr_gamma 0.95 --scheduler step --seed 42 --output_dir ./Output/Sep_15_Unet/CBSD/ConvNN_Euc_All_K9_Col_Col_S42 --data_path ./Data/CBSD68

# Loc Col 
python unet_main.py --layer ConvNN --chans 32 --num_pool_layers 4 --drop_prob 0.1 --K 9 --padding 1 --sampling_type all --num_samples -1 --shuffle_pattern NA --shuffle_scale 0 --magnitude_type cosine --similarity_type Loc --aggregation_type Col --dataset cbsd68 --noise_std 0.05 --num_epochs 50 --lr 1e-4 --lr_step 2 --lr_gamma 0.95 --scheduler step --seed 42 --output_dir ./Output/Sep_15_Unet/CBSD/ConvNN_Cos_All_K9_Loc_Col_S42 --data_path ./Data/CBSD68

python unet_main.py --layer ConvNN --chans 32 --num_pool_layers 4 --drop_prob 0.1 --K 9 --padding 1 --sampling_type all --num_samples -1 --shuffle_pattern NA --shuffle_scale 0 --magnitude_type euclidean --similarity_type Loc --aggregation_type Col --dataset cbsd68 --noise_std 0.05 --num_epochs 50 --lr 1e-4 --lr_step 2 --lr_gamma 0.95 --scheduler step --seed 42 --output_dir ./Output/Sep_15_Unet/CBSD/ConvNN_Euc_All_K9_Loc_Col_S42 --data_path ./Data/CBSD68

# Loc_Col Col 
python unet_main.py --layer ConvNN --chans 32 --num_pool_layers 4 --drop_prob 0.1 --K 9 --padding 1 --sampling_type all --num_samples -1 --shuffle_pattern NA --shuffle_scale 0 --magnitude_type cosine --similarity_type Loc_Col --aggregation_type Col --dataset cbsd68 --noise_std 0.05 --num_epochs 50 --lr 1e-4 --lr_step 2 --lr_gamma 0.95 --scheduler step --seed 42 --output_dir ./Output/Sep_15_Unet/CBSD/ConvNN_Cos_All_K9_LocCol_Col_S42 --data_path ./Data/CBSD68

python unet_main.py --layer ConvNN --chans 32 --num_pool_layers 4 --drop_prob 0.1 --K 9 --padding 1 --sampling_type all --num_samples -1 --shuffle_pattern NA --shuffle_scale 0 --magnitude_type euclidean --similarity_type Loc_Col --aggregation_type Col --dataset cbsd68 --noise_std 0.05 --num_epochs 50 --lr 1e-4 --lr_step 2 --lr_gamma 0.95 --scheduler step --seed 42 --output_dir ./Output/Sep_15_Unet/CBSD/ConvNN_Euc_All_K9_LocCol_Col_S42 --data_path ./Data/CBSD68

# Loc_Col Loc_Col 
python unet_main.py --layer ConvNN --chans 32 --num_pool_layers 4 --drop_prob 0.1 --K 9 --padding 1 --sampling_type all --num_samples -1 --shuffle_pattern NA --shuffle_scale 0 --magnitude_type cosine --similarity_type Loc_Col --aggregation_type Loc_Col --dataset cbsd68 --noise_std 0.05 --num_epochs 50 --lr 1e-4 --lr_step 2 --lr_gamma 0.95 --scheduler step --seed 42 --output_dir ./Output/Sep_15_Unet/CBSD/ConvNN_Cos_All_K9_LocCol_LocCol_S42 --data_path ./Data/CBSD68

python unet_main.py --layer ConvNN --chans 32 --num_pool_layers 4 --drop_prob 0.1 --K 9 --padding 1 --sampling_type all --num_samples -1 --shuffle_pattern NA --shuffle_scale 0 --magnitude_type euclidean --similarity_type Loc_Col --aggregation_type Loc_Col --dataset cbsd68 --noise_std 0.05 --num_epochs 50 --lr 1e-4 --lr_step 2 --lr_gamma 0.95 --scheduler step --seed 42 --output_dir ./Output/Sep_15_Unet/CBSD/ConvNN_Euc_All_K9_LocCol_LocCol_S42 --data_path ./Data/CBSD68

# Unet 
python unet_main.py --layer Conv2d --chans 32 --num_pool_layers 4 --drop_prob 0.1 --kernel_size 3 --padding 1 --dataset cbsd68 --noise_std 0.05 --num_epochs 50 --lr 1e-4 --lr_step 2 --lr_gamma 0.95 --scheduler step --seed 42 --output_dir ./Output/Sep_15_Unet/CBSD/Conv2d_K3_S42 --data_path ./Data/CBSD68
