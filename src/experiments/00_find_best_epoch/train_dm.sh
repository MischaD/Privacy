#!/bin/bash -l
#SBATCH --time=24:00:00
#SBATCH --job-name=TrainDM
#SBATCH --ntasks=1
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:8
#SBATCH -C a100_80
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
export http_proxy=http://proxy.rrze.uni-erlangen.de:80
export https_proxy=http://proxy.rrze.uni-erlangen.de:80


module load python/3.9-anaconda
module load cuda

conda activate $WORK/conda/privacy
cd $WORK/pycharm/latent-privacy

export PYTHONPATH=$PWD


accelerate launch --main_process_port=25801 ./scripts/train_unconditional.py src/experiments/base_experiment.py e00_learning_rate 