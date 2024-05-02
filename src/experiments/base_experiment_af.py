from src.experiments.base_experiment import config

config.data_csv = "cxr14privacy.csv" 
config.af_feature = "Prominent Device"

# data
config.data.limit_dataset_size = 28008#99#1770 # train data of DM 

config.af_classifier.augmix_severity = -1
config.af_classifier.random_partial_crop = 0 # deactivated for feature computation 
config.af_classifier.max_epochs=50
config.af_classifier.horizontal_flip_prop = 0
config.af_classifier.vertical_flip_prop = 0
config.af_classifier.check_val_every_n_epoch = 10 

config.af_classifier.finetune_full_model = False
config.dm_training.num_epochs = 1000 
config.dm_training.save_images_epochs = 100
config.dm_training.save_model_epochs = 100
config.dm_training.eval_fairness_epochs = 100
config.dm_training.eval_fairness = False # activate fairnesss evaluation, see config.privacy

config.dm_training.checkpointing_steps = 50000 # checkpoint steps to be used with --resume_from_checkpoint
config.dm_training.checkpoints_total_limit = 2 # != epoch saving.  