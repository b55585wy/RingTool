from enum import Enum

from nets.inception_time import InceptionTime
from nets.mamba2 import RingToolMamba
from nets.resnet import ResNet1D
from nets.transformer import RingToolBERT


class SupportedSupervisedModels(Enum):
    RESNET = "resnet"
    INCEPTION_TIME = "inception_time"
    TRANSFORMER = "transformer"
    MAMBA2 = "mamba2"


MODEL_CLASSES = {
    SupportedSupervisedModels.RESNET: ResNet1D,
    SupportedSupervisedModels.INCEPTION_TIME: InceptionTime,
    SupportedSupervisedModels.TRANSFORMER: RingToolBERT,
    SupportedSupervisedModels.MAMBA2: RingToolMamba,
}
