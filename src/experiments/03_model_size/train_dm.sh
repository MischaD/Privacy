#!/bin/bash -l
#SBATCH --time=24:00:00
#SBATCH --job-name=TrainDMModelSize
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

EXP_NAMES=("fivedown_one_lpp" "fivedown" "fourdown" "halfchannels" "onelayerperblock")
EXP_NAME=${EXP_NAMES[$SLURM_ARRAY_TASK_ID]}
EXPFILE=src/experiments/03_larger_datasets/${EXP_NAME}.py

CUR_EXP_NAME="e03_modelsize_${EXP_NAME}"

echo $CUR_EXP_NAME
echo $EXPFILE
accelerate launch --main_process_port=25801 ./scripts/train_unconditional.py ${EXPFILE} ${CUR_EXP_NAME} --data.limit_dataset_size=1770