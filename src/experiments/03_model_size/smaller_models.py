from src.experiments.base_experiment import config

config.data.limit_dataset_size = 1770

# model size
config.dm_training.layers_per_block = 2
config.dm_training.num_down_blocks = 6
config.dm_training.block_out_channels = list((128, 128, 256, 256, 512, 512))
#config.dm_training.block_out_channels = list((64, 64, 128, 128, 256, 256))

# 113_675_524  #6
# 071_439_108  #5
# 028_484_612  #4

# 028_448_388  #6 halfchannels
# 017_881_476  #5 halfchannels
# 007_135_748  #4 halfchannels

# 077_364_740  #6 one_layer_per_block
# 049_558_020  #5 one_layer_per_block
# 019_457_284  #4 one_layer_per_block