```
{
    "exp_name": "fft",
    "mode": "test",  # ["train", "test", "5fold"]
    "test_participants": "0-8",  # str: "x1-x2" or list:[x1, x2, ...]
    "device": "cpu",
    "csv_path": "csv/fft/fft.csv",
    "img_path": "img/fft",
    "method": {
        "name": "fft",
        "type": "non-ML", 
        "model_path": null,
        "params":{
            "freq_range": [0.5, 8],
            "nperseg": 128,
            "noverlap": 64,
            "nfft": 1024,
            "window": "hanning"
        }
    },
    "train":{
        "epochs": null,
        "lr": null,
        "criterion": null,
        "optimizer": null,
        "early_stopping": {
            "monitor": "val_loss",
            "patience": 10,
            "mode": "min"
        },
        "model_checkpoint": {
            "monitor": "val_loss",
            "mode": "min",
            "save_best_only": true
        }
    },
    "dataset":{
        "input_type": ["original"],
        "label_type": ["ppg","ecg"],
        "shuffle": true,
        "batch_size": 32,
        "val_ratio": 0.2
    },
    "quality_assessment":{
        "th": null,
        "method_integration": "best",
        "method_quality": "templatematch",
        "method_peaks": "elgendi"
    },
    "data_preprocess": {
        "target_fs": 60,
        "window_duration": 6,
        "hop_duration": 6,
        "test_hop":6,
        "filter": {
            "highcut": 8,
            "lowcut": 0.5,
            "order": 3
        },
        "welch": {
            "nperseg": 128,
            "noverlap": 64,
            "nfft": 1024,
            "window": "hanning"
        },
        "file_path": "TODO", 
        "skip_subjects":["2", "13", "16"],
        "task_oi": [
            "TODO"
            ],
        "methods": ["original", "difference", "frequency"],
        "selected_features":{
            "ring": [
                "PPG_Red", 
                "PPG_IR", 
                "PPG_Green",
                "ACC_X",
                "ACC_Y",
                "ACC_Z"
            ]
        }
    }
}
```