#!/bin/bash

CVD=(2 3)
EXPNAMES=("train_saf_classifier" "train_af_classifier")
CMDS=("--use_synthetic_af" "")


for i in "${!CVD[@]}"; do
    echo "Start session: session${EXPNAMES[$i]}"
    screen -dmS "session${EXPNAMES[$i]}" -L -Logfile "session${EXPNAMES[$i]}_log.txt" bash -c "\
    cd /vol/ideadata/ed52egek/pycharm/latent-privacy; export CUDA_VISIBLE_DEVICES=${CVD[$i]}; export PYTHONPATH=/vol/ideadata/ed52egek/pycharm/latent-privacy;\
    /vol/ideadata/ed52egek/conda/latentprivacy/bin/python /vol/ideadata/ed52egek/pycharm/latent-privacy/scripts/sweep_af_classifier.py \
    src/experiments/base_experiment.py \
    ${EXPNAMES[$i]} \
    ${CMDS[$i]} \
    --n_sweeps=10 \
    --data_csv=cxr14supportdevices.csv
    "
done


#conda activate $CONDA/latentprivacy; export CUDA_VISIBLE_DEVICES=1; export PYTHONPATH=/vol/ideadata/ed52egek/pycharm/latent-privacy; python scripts/sweep_id_classifier.py src/experiments/base_experiment.py train_id_classifier --use_synthetic_af --n_sweeps=20 --data_csv=cxr14supportdevices.csv
#conda activate $CONDA/latentprivacy; export CUDA_VISIBLE_DEVICES=1; export PYTHONPATH=/vol/ideadata/ed52egek/pycharm/latent-privacy; python scripts/sweep_id_classifier.py src/experiments/05_sunglasses/find_sunglasses.py train_id_classifier --n_sweeps=20 --data_csv=celebahq_latent.csv
echo "Screen sessions have been started"