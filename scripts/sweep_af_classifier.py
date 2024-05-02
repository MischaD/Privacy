import os
import wandb
from train_af_classifier import main as main_train_classifier
from utils import main_setup
from log import logger
import argparse


def main(config): 
    wandb.login()

    sweep_configuration = {
        "name": str(config.EXP_NAME),
        "method": "random",
        "metric": {"goal": "maximize", "name": "test_acc"},
        "parameters": {
            "af_classifier.lr": {"max": 0.03, "min": 0.0005},
            "af_classifier.augmix_severity": {"values": [-1,]},
            "af_classifier.learning_rate_annealing_patience": {"values": [3, 5, 10]},
            "af_classifier.max_epochs": {"values": [10, 30, 50, 100]}, # shorter traininig for ret - todo include in config file 
            "af_classifier.gaussian_blur": {"values": [True, False]},
        },
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project=f"{config.EXP_NAME}")
    logger.info(f"Running sweep for {config.n_sweeps} times")
    wandb.agent(sweep_id, function=main_sweep_classifier, count=config.n_sweeps)


def main_sweep_classifier():
    wandb.init(project="privacy")
    for k, v in wandb.config.items(): 
        # automate overwriting of config.x.y.z = wandb.config.'x.y.z' 
        attr_keys = k.split(".")
        updated_config = v
        while attr_keys != []:
            updated_config = {attr_keys.pop(-1): updated_config}
        config.update(updated_config)
    score = main_train_classifier(config)
    wandb.log({"test_acc": score})
    return score

def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("EXP_PATH", type=str, help="Path to experiment file")
    parser.add_argument("EXP_NAME", type=str, help="Path to Experiment results")
    parser.add_argument("--data_csv", type=str, help="data dir or csv with paths to data")
    parser.add_argument("--n_sweeps", type=int, help="number of sweeps to perform")
    parser.add_argument("--tags", type=str, default="", help="wandb tags")
    parser.add_argument("--use_synthetic_af",  action='store_true')
    parser.add_argument("--af_classifier.finetune_full_model",  action='store_true')

    #parser.add_argument("--af_inpainter_name", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    config = main_setup(args, name=os.path.basename(__file__).rstrip(".py"))
    config.tags = "sweep"
    main(config)
