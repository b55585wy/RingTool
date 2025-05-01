# How to write a config file
RingTool uses a JSON file to configure the training and testing of the model. The configuration file is divided into several sections, each with its own purpose. Below is an example of a configuration file for a ResNet model.

## Configuration File Structure
The configuration file is divided into several sections, each with its own purpose. Below is a description of each section:

**`exp_name`**: The name of the experiment. This should be a unique name that describes the experiment. If you are experimenting with different configurations, you can use a naming convention to keep track of them. The same `exp_name` by default will overwrite the existing models so be careful when using the same name.

**`mode`**: The mode of the experiment. This can be either `5fold`, `train`, or `test`. In "5fold" mode, the data is split into 5 folds for cross-validation. In "train" mode, the model is trained on the `split.train`, validated on the `split.valid` and tested on `split.test`. In "test" mode, RingTool will repeat the 5fold testing process.

**`dataset`**: This section contains the configuration for the dataset. It includes the following parameters:
- `ring_type`: The type of ring data to be used. This can be either `ring1` or `ring2`.
- `input_type`: The input channels to be used, which can be chose from `["ir-raw","ir-filtered","ir-standardized","ir-difference","ir-welch","ir-filtered-rr","ir-welch-rr","red-raw","red-filtered","red-standardized","red-difference","red-welch","red-filtered-rr","red-welch-rr"."ax-raw","ax-filtered","ax-standardized","ax-difference","ax-welch","ax-filtered-rr","ax-welch-rr","ay-raw","ay-filtered","ay-standardized","ay-difference","ay-welch","ay-filtered-rr","ay-welch-rr","az-raw","az-filtered","az-standardized","az-difference","az-welch","az-filtered-rr","az-welch-rr"]`. We recommend using `["ir-filtered"]` for the HR task and `["ir-filtered-rr"]` for the RR task.
- `label_type`: The type of label in the dataset to be used, which can be chose from `["hr", "spo2", "bvp_sdnn","resp_rr","samsung_hr","oura_hr","BP_sys","BP_dia"]`.
- `shuffle`: Whether to shuffle the dataset.
- `batch_size`: The batch size to be used for training and testing.
- `quality_assessment`: The quality assessment method to be used. This can be either `elgendi` or `none`. If you choose `elgendi`, you need to set the threshold (`th`) for the quality assessment. The default value is 0.
- `target_fs`: The target sampling frequency to be used. This is the frequency at which the data will be resampled.
- `window_duration`: The duration of the window to be used for dataset augmentation. See [load_dataset.py](../dataset/load_dataset.py) for more details.
- `experiment`: The type of experiment to be used. Default is `["Health", "Daily", "Sport"]`.
- `task`: The scenarios to be loaded from the dataset. You can choose from `["sitting", "spo2", "deepsquat", "talking", "shaking_head", "standing", "striding"]`. We recommend using the **universal set** for training, and the specific set for testing. For example, if you want to experiment the motion scenarios, you can set the `task` to `["sitting", "spo2", "deepsquat", "talking", "shaking_head", "standing", "striding"]` for training and `["deepsquat", "striding"]` for testing.
- `accel_combined`: Whether to combine the accelerometer data. This is only used for the `ax`, `ay`, and `az` channels. If set to `true`, the accelerometer data will be combined using the method specified in `accel_combined_method`.
- `accel_combined_method`: The method to be used for combining the accelerometer data. Data to be chosen from `["magnitude", "sma", "rms", "jerk", "pitch", "roll", "pitch_deg", "roll_deg"]`. If you set `accel_combined` to `false`, this parameter will be ignored. You can refer to [accel_features.py](../utils/accel_features.py) for more details.

**`seed`**: The random seed to be used for training and testing. This is used to ensure that the results are reproducible. The default value is 42.

**`csv_path`**: The path to the output CSV files. The current logic doesn't allow customization yet.

**`img_path`**: The path to the directory where the images will be saved. The current logic doesn't allow customization yet.

**`method`**: This section contains the configuration for the supervised learning method. For `name` field, you can choose from `["resnet", "transformer", "mamba2", "inception_time"]`. For `type` field, you can choose from `["ML", "unsupervised"]`. The `model_path` is the path to the pre-trained model. The `params` section contains the parameters for model initialization. Please refer to the model's `__init__(self)` function for detailed parameters. The `params` section is different for each model. For now, RingTool will not check the correctness of the parameters. So please make sure that the parameters are well matched with the model.

**`train`**: This section contains the configuration for training the model. It includes the following parameters:
- `device`: The device to be used for training. If you have multiple GPUs, you can set this to the GPU ID you want to use. It can also be set to `cpu` but we don't recommend it.
- `epochs`: The number of epochs to be used for training. The default value is 200. This is good enough for RingTool.
- `lr`: The learning rate to be used for training. The default value is 1e-3.
- `criterion`: The loss function to be used for training. The default value is `mse`. You can also choose from `["mae", "mse", "cross_entropy"]`.
- `optimizer`: The optimizer to be used for training. The default value is `adam`. You can also choose from `["adam", "adamw"]`.
- `early_stopping`: The early stopping method to be used. This is used to stop the training process if the validation loss does not improve for a certain number of epochs. The `monitor` parameter specifies the metric to be monitored, the `patience` parameter specifies the number of epochs to wait before stopping, and the `mode` parameter specifies whether to minimize or maximize the metric.
- `scheduler`: The learning rate scheduler to be used. The `type` parameter specifies the type of scheduler to be used. The `factor` parameter specifies the factor by which to reduce the learning rate, the `patience` parameter specifies the number of epochs to wait before reducing the learning rate, the `threshold` parameter specifies the minimum change in the metric to be considered an improvement, and the `min_lr` parameter specifies the minimum learning rate to be used.
- `model_checkpoint`: The model checkpointing method to be used. This is used to save the model at the end of each epoch. The `monitor` parameter specifies the metric to be monitored, the `mode` parameter specifies whether to minimize or maximize the metric, and the `save_best_only` parameter specifies whether to save only the best model.

**`test`**: This section contains the configuration for testing the model. It includes the following parameters:
- `device`: The device to be used for testing. If you have multiple GPUs, you can set this to the GPU ID you want to use. It can also be set to `cpu`.
- `batch_size`: The batch size to be used for testing.
- `metrics`: The metrics to be used for testing. You can choose from `["mae", "rmse", "mape", "pearson"]`.
- `model_path`: The path to the model to be used for testing. If you leave this as `null`, the model will be loaded from the validation checkpoint.
- `model_name`: Not implemented yet.
- `pretrain_model`: Not implemented yet.

## Example Configuration File
```json
{
    "exp_name": "resnet-ring1-hr-all-ir",
    "mode": "5fold",
    "split":{
        "train": ["00009", "00012", "00005", "00020", "00031", "00022", "00029", "00016", "00026", "00024", "00014", "00010", "00011", "00027", "00008", "00019", "00030", "00003", "00025", "00006", "00033"], 
        "valid": ["00013", "00018", "00002", "00032", "00028", "00021", "00000"], 
        "test": ["00023", "00004", "00040", "00015", "00017", "00001", "00007"], 
        "5-Fold": {"Fold-1": ["00023", "00004", "00040", "00015", "00017", "00001", "00007"], 
            "Fold-2": ["00013", "00018", "00002", "00032", "00028", "00021", "00000"], 
            "Fold-3": ["00009", "00012", "00005", "00020", "00031", "00022", "00029"],
            "Fold-4": ["00016","00025", "00024", "00014", "00010", "00011", "00027"], 
            "Fold-5": ["00008", "00019", "00030", "00003", "00026" , "00006", "00033"]
    }
    },
    "dataset":{
        "ring_type": "ring1",
        "input_type": ["ir-filtered"],
        "label_type": ["hr"],
        "shuffle": true,
        "batch_size": 128,
        "quality_assessment": {
            "method": "elgendi",
            "th": 0
        },
        "target_fs": 100,
        "window_duration": 30,
        "experiment": ["Health", "Daily", "Sport"],
        "task": ["sitting", "spo2", "deepsquat", "talking", "shaking_head", "standing", "striding"],
        "accel_combined": false,
        "accel_combined_method": "magnitude"
    },
    "seed": 42,
    "csv_path": "csv/resnet/resnet.csv",
    "img_path": "img/resnet",
    "method": {
        "name": "resnet",
        "type": "ML", 
        "model_path": null,
        "params":{
            "in_channels": 1,  
            "base_filters": 32,
            "kernel_size": 5,
            "stride": 1,
            "groups": 1,
            "n_block": 8,
            "downsample_gap": 2,
            "increasefilter_gap": 2,
            "use_do": true,
            "dropout_p": 0.3,
            "use_final_do": false,
            "final_dropout_p": 0.5,
            "backbone": false
        }
    },
    "train":{
        "device": "0",
        "epochs": 200,
        "lr": 1e-3,
        "criterion": "mse",
        "optimizer": "adam",
        "early_stopping": {
            "monitor": "val_loss",
            "patience": 200,
            "mode": "min"
        },
        "scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "patience": 10,
            "threshold": 1e-4,
            "min_lr": 1e-6
        },
        "model_checkpoint": {
            "monitor": "val_loss",
            "mode": "min",
            "save_best_only": true
        }
    },
    "test":{
        "device": "0",
        "batch_size": 128,
        "metrics": ["mae", "rmse", "mape", "pearson"],
        "model_path": null,
        "model_name": null
    },
    "pretrain_model": "TODO"
}
```

## Future Work
Currently RingTool uses JSON for parsing arguments and hyperparameter. In the future we might integrate Hydra and Omegaconf for better management. PRs are welcome!
