import ml_collections
import os
import torch
import math


config = ml_collections.ConfigDict()
config.seed = 42
config.base_dir = "/vol/ideadata/ed52egek/data/" if os.path.abspath(".").startswith("/vol") else "/home/atuin/b180dc/b180dc10/data/"

# data
config.data = data = ml_collections.ConfigDict()
config.data.dataset_shuffle_seed = 10
config.data.image_size = 512

# saf 
saf = config.data.saf = ml_collections.ConfigDict()
saf.training_data_probability = 0.5
saf.radius = int(9 * 2 ** (math.log2(config.data.image_size) - 6))# 72 for 512 images
saf.seed = 30

# af classifer
config.af_classifier = af_classifier = ml_collections.ConfigDict()
af_classifier.early_stopping = "val_loss"
af_classifier.lr = 1e-3
af_classifier.augmix_severity = 3
af_classifier.gaussian_blur = False
af_classifier.random_partial_crop = 0.5 # deactivated for feature computation 
af_classifier.batch_size = 32 
af_classifier.augment_private_images_to_balance = True
af_classifier.max_epochs =50 
af_classifier.horizontal_flip_prop = 0.5
af_classifier.vertical_flip_prop = 0.5
af_classifier.num_workers = 15
af_classifier.check_val_every_n_epoch = 5 # also used for learning rate annealing
af_classifier.learning_rate_annealing = True 
af_classifier.learning_rate_annealing_patience = 3 # only checked every af_classifier.check_val_every_n_epoch 
af_classifier.best_path = f"af_{config.af_inpainter_name}_classifier_last.ckpt" # in EXPERIMENT_NAME
af_classifier.mask_only = False # ignored if pii 

## DM training -- more documentation at https://github.com/huggingface/diffusers/blob/main/examples/unconditional_image_generation/train_unconditional.py
config.dm_training = dm_training = ml_collections.ConfigDict()
dm_training.resolution = 64  
dm_training.center_crop = True
dm_training.random_flip = False # horizontally
dm_training.train_batch_size = 16
dm_training.eval_batch_size = 16
dm_training.num_epochs = 1000
dm_training.save_images_epochs = 100
dm_training.save_model_epochs = 100
dm_training.eval_fairness_epochs = 100
dm_training.eval_fairness = False # activate fairnesss evaluation, see config.privacy
dm_training.gradient_accumulation_steps = 1
dm_training.learning_rate = 1e-4
dm_training.lr_scheduler = "constant" # "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"
dm_training.lr_warmup_steps = 500
dm_training.adam_beta1 = 0.95
dm_training.adam_beta2 = 0.999
dm_training.adam_weight_decay = 1e-6
dm_training.adam_epsilon = 1e-08
dm_training.use_ema = True 
dm_training.ema_inv_gamma = 1.0
dm_training.ema_power = 3 /4
dm_training.ema_max_decay = 0.9999
dm_training.mixed_precision = "no" # "fp16, bf16"
dm_training.prediction_type = "epsilon" # "epsilon, sample"
dm_training.ddpm_num_steps = 1000
dm_training.ddpm_num_inference_steps = 100
dm_training.ddpm_beta_schedule = "linear"
dm_training.ddpm_beta_end = 0.02
dm_training.layers_per_block = 2
dm_training.num_down_blocks = 6
dm_training.checkpointing_steps = 50000 # checkpoint steps to be used with --resume_from_checkpoint
dm_training.checkpoints_total_limit = 2 # != epoch saving.  

config.privacy = privacy = ml_collections.ConfigDict()
privacy.online_M = 16 # number for prediction for each t timestep 
privacy.memorized_t_threshold = 0.70 # if t is higher than this, samples are memorized 
privacy.forgotten_t_threshold = 0.43 # if t is lower than this, samples are forgotton  
privacy.always_compute_forgotten = True # if False skips computing forgotten if images are memorized and sets all vals to M
privacy.start_epoch = 200
privacy.evaluation_M = 16
privacy.evaluation_step_size = 0.01 # at sampling time 
privacy.cooldown_epochs = 0 # after a change - how long do you want to wait 

#
## sampling
config.sampling = sampling = ml_collections.ConfigDict()
sampling.N = 50000
sampling.ddpm_inference_steps = 100 

config.evaluate = evaluate = ml_collections.ConfigDict()
evaluate.number_of_synth_samples_for_psnr = 1000
evaluate.fid_n_bootstrap = 10 # how often
evaluate.bootstrap_threshold = 5000
