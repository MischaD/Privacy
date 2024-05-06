#!/bin/bash -l
#SBATCH --time=24:00:00
#SBATCH --job-name=TrainDMDSSizeLarger
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

echo ${SLURM_ARRAY_TASK_ID}

conda activate $WORK/conda/privacy
cd $WORK/pycharm/latent-privacy
export PYTHONPATH=$PWD

EXPFILE=src/experiments/02_larger_datasets/large_datasets.py
EXPPATHS=(875 1750 3500 7001 14003 28007)
CUR_EXP_NAME="e02_dssize_${EXPPATHS[$SLURM_ARRAY_TASK_ID]}"

echo $CUR_EXP_NAME

accelerate launch --main_process_port=25801 ./scripts/train_unconditional.py ${EXPFILE} ${CUR_EXP_NAME} --data.limit_dataset_size=${EXPPATHS[$SLURM_ARRAY_TASK_ID]} --use_synthetic_af