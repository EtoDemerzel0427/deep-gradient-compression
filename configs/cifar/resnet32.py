from torchpack.mtpack.models.vision.resnet import resnet32

from torchpack.mtpack.utils.config import Config, configs

# model
configs.model = Config(resnet32)
configs.model.num_classes = configs.dataset.num_classes