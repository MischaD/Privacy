#!/bin/bash -l
#SBATCH --time=06:00:00
#SBATCH --job-name=TestDM
#SBATCH --ntasks=1
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
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
EXPFILE=src/experiments/base_experiment.py
EXPNAME=e00_learning_rate

if [[ $SLURM_ARRAY_TASK_ID -eq 4 ]]; then 
    EPOCH_NUM=final
else
    EPOCH_NUM=$(( (SLURM_ARRAY_TASK_ID + 1) * 10 ))
    EPOCH_NUM=$(printf "%05d" ${EPOCH_NUM}000)
    EPOCH_NUM=epoch-${EPOCH_NUM}
fi

python ./scripts/sample.py $EXPFILE $EXPNAME  --model_dir=$EPOCH_NUM
python ./scripts/compute_fid.py $EXPFILE $EXPNAME --samples_path=log/$EXPNAME/${EPOCH_NUM}/samples