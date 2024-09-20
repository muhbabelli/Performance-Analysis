#!/bin/bash

#SBATCH --job-name=cityscapes_gmmseg_inference      
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=5
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:tesla_t4:1
#SBATCH --time=2:00:00
#SBATCH --output=inference-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mbabelli22@ku.edu.tr
#SBATCH -e=error.err
#SBATCH --mem=25000



module load anaconda/3.6
module load cuda/11.8.0
eval "$(conda shell.bash hook)"
conda activate deeplearning

export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'

python inference.py --ckpt /kuacc/users/mbabelli22/myworkfolder/Project2_Performance_Analysis/Gmmseg/semantic-segmentation/semantic-segmentation/ckpt/epoch=236-val_iou=0.83.ckpt --cityscapes_root '/datasets/cityscapes/'