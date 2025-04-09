import os
from nets.resnet import ResNet1D


def load_model(config):
    if config["name"] == "resnet":
        return ResNet1D(
            **config["params"]
        )