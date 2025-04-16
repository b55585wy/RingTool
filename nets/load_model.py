import os
from nets.resnet import ResNet1D
from nets.transformer import MyTransformer
# from nets.inception import Inception

def load_model(config):
    if config["name"] == "resnet":
        return ResNet1D(
            **config["params"]
        )
    elif config['name'] == "transformer":

        return MyTransformer(
            **config["params"]
        )
    