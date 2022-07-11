from torchpack.mtpack.utils.config import Config, configs

from dgc.new_compression import RawTopKCompressor
from dgc.memory import DGCSGDMemory
from dgc.optim import DGCSGD


configs.train.dgc = True
configs.train.compression = Config(RawTopKCompressor)
configs.train.compression.compress_ratio = 0.003

old_optimizer = configs.train.optimizer
configs.train.optimizer = Config(DGCSGD)
for k, v in old_optimizer.items():
    configs.train.optimizer[k] = v

configs.train.compression.memory = Config(DGCSGDMemory)
configs.train.compression.memory.momentum = configs.train.optimizer.momentum