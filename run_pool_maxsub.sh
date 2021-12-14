#!/bin/bash
#SBATCH --mail-type=END
#SBATCH --mail-user=ma.xu1@northeastern.edu
#SBATCH -N 1
#SBATCH -p multigpu
#SBATCH --gres=gpu:v100-sxm2:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128Gb
#SBATCH --time=1-00:00:00
#SBATCH --output=%j.log

source activate pt
cd /scratch/ma.xu1/point-transformer
sh train.sh s3dis pointtransformer_pool_maxsub
