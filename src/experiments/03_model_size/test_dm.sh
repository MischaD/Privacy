#!/bin/bash -l
#SBATCH --time=12:00:00
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
EXP_NAMES=("fivedown_one_lpp" "fivedown" "fourdown" "halfchannels" "onelayerperblock")
EXP_NAME=${EXP_NAMES[$SLURM_ARRAY_TASK_ID]}
EXPFILE=src/experiments/03_model_size/${EXP_NAME}.py
           
python ./scripts/sample.py $EXPFILE $EXP_NAME --model_dir=final
python ./scripts/compute_fid.py $EXPFILE $EXP_NAME --samples_path=log/${EXP_NAME}/final/samples
python ./scripts/test_model.py $EXPFILE $EXP_NAME --samples_path=log/${EXP_NAME}/final/samples --af_classifier_path=log/af_best.ckpt --id_classifier_path=log/id_best.ckpt
python ./scripts/compute_t_dash.py $EXPFILE $EXP_NAME --use_synthetic_af --model_dir=final --af_classifier_path=log/af_dev.ckpt --id_classifier_path=log/id_best.ckpt