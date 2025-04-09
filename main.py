import json
import os
import numpy as np
import pandas as pd
from scipy.signal import welch, get_window, butter, filtfilt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import argparse
from nets.load_model import load_model
from dataset.load_dataset import load_dataset
from trainer.load_trainer import load_trainer
from typing import List, Dict, Optional, Union
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random


DATA_PATH = "/home/disk2/disk/3/tjk/RingData/ring_data.npy"


def generate_split_config(mode: str, all_p: List, test_p: Optional[Union[List, str]] = None):
    split_config = []
    if mode == "5fold":
        # generate [{train:[], test:[]}*5] as split config
        pass
    else:
        # TODO: all_p may not be List of int.
        if test_p is None:
            raise ValueError("test_participants must be provided for train mode or test mode.")
        if isinstance(test_p, str):
            test_p = range(int(test_p.split("-")[0]), int(test_p.split("-")[1]))
        test_p = ["{:05d}".format(p) for p in test_p]
        if mode == "train":
            train_p = [p for p in all_p if p not in test_p]
            random.shuffle(train_p)
            valid_p = train_p[:len(train_p)//4]
            train_p = train_p[len(train_p)//4:]
            split_config.append({"train": train_p, "test": test_p, "valid": valid_p})
        else:
            split_config.append({"test": test_p})
    return split_config


    
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config




def main(config_path):
    config = load_config(config_path)
    
    # load all data
    all_data = np.load(DATA_PATH, allow_pickle=True).item()
    # training 
    if config["mode"] not in ["train", "test", "5fold"]:
        raise ValueError("Invalid mode. Choose 'train' or 'test', '5fold'.")
    split_configs = generate_split_config(config["mode"], list(all_data.keys()), config["test_participants"])
    
    results = []
    for split_config in split_configs:
        # load model
        model = load_model(config['method'])
        print(f"Running experiment with split config: {split_config}")
        trainer = load_trainer(model, config['method']['name'], config["train"])
        if "train" in split_config:
            # prepare training dataset
            train_data = [all_data[p] for p in split_config["train"]]
            train_dataset = load_dataset(config["data_preprocess"], train_data)
            train_loader = DataLoader(train_dataset, batch_size=config["dataset"]["batch_size"], shuffle=True)
            valid_data = [all_data[p] for p in split_config["valid"]]
            valid_dataset = load_dataset(config["data_preprocess"], valid_data)
            valid_loader = DataLoader(valid_dataset, batch_size=config["dataset"]["batch_size"], shuffle=False)
            # TODO: does all methods requires both training set and valid set? yes and the split is 3:1:1
            trainer.fit(train_loader, valid_loader)
        # test model 
        test_data = [all_data[p] for p in split_config["test"]]
        test_dataset = load_dataset(config["data_preprocess"], test_data)
        test_loader = DataLoader(test_dataset, batch_size=config["dataset"]["batch_size"], shuffle=False)
        test_results = trainer.test(test_loader)
        results.append(test_results)
    
    # TODO: save results or show results, different task have different results.
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process ring PPG data using FFT.')
    parser.add_argument('--config', type=str, default="./config/Resnet.json", help='Path to the configuration JSON file.')
    args = parser.parse_args()
    
    main(args.config)
