import argparse
import datetime
import json
import logging
import os
import random
import time
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from constants.experiment import ExperimentMode
from dataset.load_dataset import DatasetType, load_dataset
from nets.load_model import load_model
from notifications.slack import (
    format_results_to_slack_blocks,
    send_slack_message,
    setup_slack,
)
from trainer.load_trainer import load_trainer
from utils.utils import calculate_avg_metrics, save_metrics_to_csv, save_prediction_pairs_detailed


def generate_split_config(mode: str, split: Dict) -> List[Dict]:
    split_config = []
    # 5-fold cross-validation.
    # if test set is fold 4, then valid set is fold 5 and train set is 1, 2, 3 train set is fold 1, 2, 3
    if mode == ExperimentMode.FIVE_FOLD.value or mode == ExperimentMode.TEST.value:
        for i in range(5):
            test_fold = i + 1  # Folds are 1-indexed
            valid_fold = (i + 1) % 5 + 1  # Wraps around to fold 1 after fold 5

            valid_p = split['5-Fold'][f'Fold-{valid_fold}']
            test_p = split['5-Fold'][f'Fold-{test_fold}']
            
            # Train participants are from the remaining folds
            train_p = []
            for j in range(1, 6):  # Folds are 1-indexed
                if j != valid_fold and j != test_fold:
                    train_p.extend(split['5-Fold'][f'Fold-{j}'])
            
            split_config.append({"train": train_p, "valid": valid_p, "test": test_p, "fold": f"Fold-{test_fold}"})
    elif mode == ExperimentMode.TRAIN.value:
        # split into train, valid, test
        split_config.append({"train": split['train'], "valid": split['valid'], "test": split['test'], "fold": "Fold-1"})
    
    else:
        logging.error(f"Invalid mode. Choose from {[mode.value for mode in ExperimentMode]}.")
        raise ValueError(f"Invalid mode. Choose from {[mode.value for mode in ExperimentMode]}.")
    return split_config


def load_config(config_path: str) -> Dict:
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config


def find_all_data(path, ring_type) -> Dict[str, pd.DataFrame]:
    # load all subject data from a folder, subject_ring1_processed.pkl
    all_data = {}  # subject_id -> pd.DF
    for filename in os.listdir(path):
        if filename.endswith('.pkl') and ring_type in filename:
            # load data
            file_path = os.path.join(path, filename)
            try:
                data = pd.read_pickle(file_path)
                # get subject id from filename
                subject_id = filename.split('_')[0]
                # add data to dictionary
                all_data[subject_id] = data
            except Exception as e:
                logging.error(f"Error loading {filename}: {e}")
                continue
    return all_data


def set_seed(seed: int):
    """Set the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unsupervised(config: Dict, data_path: str) -> None:   
    # load all data
    all_data = find_all_data(data_path, config["dataset"]["ring_type"])
    subject_list = list(all_data.keys())
    all_data = pd.concat(all_data.values())
    logging.info(f"Found {len(subject_list)} subjects in the data folder.") 
    # set seed
    set_seed(config["seed"])
    # only test on the whole dataset without split, unsupervised methods
    if config["mode"] not in [mode.value for mode in ExperimentMode]:
        logging.error(f"Invalid mode: {config['mode']}. Choose from {[mode.value for mode in ExperimentMode]}.")
        raise ValueError(f"Invalid mode. Choose from {[mode.value for mode in ExperimentMode]}.")
    if config["mode"] == ExperimentMode.TEST.value and config["method"]["type"]== "unsupervised":
        # load dataset
        channels = config["dataset"]["input_type"]
        tasks = config["dataset"]["label_type"]
        logging.info(f"Channels: {channels}, Task: {tasks}")
        tester = load_trainer(config['method'], config['method']['name'], config)
        for task in tasks:
            all_dataset = load_dataset(
                config=config,
                raw_data=all_data,
                channels=channels,
                task=task,
                scenarios=config["dataset"]["task"]
            )
            all_loader = DataLoader(all_dataset, batch_size=config["dataset"]["batch_size"], shuffle=False)
            test_results = tester.test(all_loader, None, task)
            # print test results
            logging.info(f"Test results for task {task}: {test_results}")


def supervised(config: Dict, data_path: str) -> List[Tuple[str, str, Dict]]:
    mode = config["mode"]  # "train", "test", "5fold"
    exp_name = config.get("exp_name")
    all_data = find_all_data(data_path, config["dataset"]["ring_type"])
    subject_list = list(all_data.keys())

    logging.info(f"Found {len(subject_list)} subjects in the data folder.")
    # set seed
    set_seed(config["seed"])
    # training 
    if mode not in [mode.value for mode in ExperimentMode]:
        logging.error(f"Invalid mode: {mode}. Choose from {[mode.value for mode in ExperimentMode]}.")
        raise ValueError(f"Invalid mode. Choose from {[mode.value for mode in ExperimentMode]}.")
    # check if the key in split_config is in the subject list, save the cross into split_config
    # Correct the call to generate_split_config
    split = config.get("split", {})
    split_configs = generate_split_config(mode, split)
    
    # Check if all subjects in split_configs exist in available data
    for split_config in split_configs:
        for split_type in ["train", "valid", "test"]:
            if split_type in split_config:
                # Filter out subjects that don't exist in available data
                split_config[split_type] = [subj for subj in split_config[split_type] if subj in subject_list]
    logging.info(f"Generated {len(split_configs)} split configurations.")
 
    all_test_results = []
    tasks = config["dataset"]["label_type"]
    for task in tasks:
        logging.info(f"Running experiment for task: {task}")
        all_preds_and_targets: List[Tuple] = []
        # Extract channels and task from config
        channels = config["dataset"]["input_type"]
        logging.info(f"Channels: {channels}, Task: {tasks}")
        for split_config in split_configs:
            current_fold = split_config["fold"]
            config["fold"] = current_fold  # TODO: remove dynamic config setter

            checkpoint_path = None
            # for testing, use the checkpoint path
            if mode == ExperimentMode.TEST.value:
                if config.get("test", {}).get("model_path"):
                    checkpoint_path = config["test"]["model_path"]
                    logging.info(f"Using checkpoint path from config model_path: {checkpoint_path}")
                else:
                    checkpoint_subdir = "hr" if task in ["samsung_hr", "oura_hr"] else task
                    
                    # example: exp_name: inception-time-ring1-samsung_hr-motion-ir -> inception-time-ring1-hr-all-ir
                    if "motion" in exp_name:
                        exp_name_subdir = exp_name.replace("motion", "all")
                    elif "stationary" in exp_name:
                        exp_name_subdir = exp_name.replace("stationary", "all")
                    else:
                        exp_name_subdir = exp_name
                    if "samsung_hr" in exp_name_subdir:
                        exp_name_subdir = exp_name_subdir.replace("samsung_hr", "hr")
                    elif "oura_hr" in exp_name_subdir:
                        exp_name_subdir = exp_name_subdir.replace("oura_hr", "hr")
                    

                    checkpoint_dir = os.path.join("models", exp_name_subdir, checkpoint_subdir, current_fold)
                    checkpoint_path = os.path.join(checkpoint_dir, f"{exp_name_subdir}_{checkpoint_subdir}_{current_fold}_best.pt")
                    logging.info(f"Using checkpoint path from default setting. checkpoint_path: {checkpoint_path}")
                if not os.path.exists(checkpoint_path):
                    logging.error(f"Checkpoint {checkpoint_path} not found. Maybe you need to train the model first.")
                    raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found. Maybe you need to train the model first.")


            logging.info(f"Now running experiment {current_fold} with split config: {split_config}")      
            # load model
            model = load_model(config['method'])
            logging.info(f"Successfully loaded model {config['method']}")
            logging.info(f"Model params: {sum(p.numel() for p in model.parameters())}")
            logging.info(f"Running experiment with split config: {split_config}")

            trainer = load_trainer(model, config['method']['name'], config)
            
            train_task = "hr" if task in ["oura_hr", "samsung_hr"] else task

            if "train" in split_config and (mode == ExperimentMode.TRAIN.value or mode == ExperimentMode.FIVE_FOLD.value):
                # prepare training dataset
                train_data = pd.concat([all_data[p] for p in split_config["train"]])
                train_dataset = load_dataset(
                    config=config,
                    raw_data=train_data,
                    channels=channels,
                    task=train_task,
                    dataset_type=DatasetType.TRAIN
                )
                train_loader = DataLoader(train_dataset, batch_size=config["dataset"]["batch_size"], shuffle=True)
                
                valid_data = pd.concat([all_data[p] for p in split_config["valid"]])
                valid_dataset = load_dataset(
                    config=config,
                    raw_data=valid_data,
                    channels=channels,
                    task=task,
                    dataset_type=DatasetType.VALID
                )
                valid_loader = DataLoader(valid_dataset, batch_size=config["dataset"]["batch_size"], shuffle=False)
                
                # Train the model
                checkpoint_path, config_path = trainer.fit(train_loader, valid_loader, task, current_fold)
                logging.info(f"Model trained and saved to {checkpoint_path}.")
                logging.info(f"Model config saved to {config_path}.")
        
            # test model 
            test_data = pd.concat([all_data[p] for p in split_config["test"]])
            test_dataset = load_dataset(
                config=config,
                raw_data=test_data,
                channels=channels,
                task=task,
                dataset_type=DatasetType.TEST,
                scenarios=config["dataset"]["task"],  # TODO: naming issue
            )
            test_loader = DataLoader(test_dataset, batch_size=config["dataset"]["batch_size"], shuffle=False)
            test_results = trainer.test(test_loader, checkpoint_path, task)
            preds_and_targets = test_results["preds_and_targets"]
            all_preds_and_targets.append(preds_and_targets)

            all_test_results.append((split_config["fold"], task, test_results))
            
            # Save prediction pairs when --save-predictions flag is enabled
            if config.get('_save_predictions_', False):
                exp_name = config.get("exp_name", "unknown")
                csv_path = os.path.join("predictions", exp_name, f"{split_config['fold']}.csv")
                preds, targets = preds_and_targets
                save_prediction_pairs_detailed(
                    preds=preds,
                    targets=targets,
                    save_path=csv_path,
                    metadata=None,  # main.py doesn't collect metadata
                    task=task,
                    fold=split_config["fold"],
                    exp_name=exp_name
                )

        metrics = calculate_avg_metrics(all_preds_and_targets)
        logging.critical(f"Average metrics across all tasks: "
                f"MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}, "
                f"MAPE: {metrics['mape']:.2f}%, Pearson: {metrics['pearson']:.4f}")

        # Save overall metrics to CSV
        config["fold"] = "all-folds"  # TODO: remove dynamic config setter
        save_metrics_to_csv(metrics, config, task)
        # # Plot and save metrics
        # plot_and_save_metrics(
        #     predictions=torch.cat([p_and_t[0] for p_and_t in all_preds_and_targets]),
        #     targets=torch.cat([p_and_t[1] for p_and_t in all_preds_and_targets]),
        #     config=config,
        #     task=task,
        # )

    return all_test_results


def setup_logging(exp_name: str, config: Dict) -> None:
    # Set up logging
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%m%d-%H%M%S")
    log_filename = f"logs/rtool-{exp_name}-{timestamp}.log"

    # Remove existing handlers if any, to avoid duplicate logs when running multiple configs
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    logging.getLogger('matplotlib').setLevel(logging.INFO) # Reduce matplotlib verbosity
    logging.info(f"Starting experiment: {exp_name}.")
    logging.debug(f"Config: {json.dumps(config, indent=2)}")
    logging.info(f"Logging to: {log_filename}")


def do_run_experiment(config: Dict, data_path: str, send_notification_slack=False):
    """
    Run the experiment based on the provided configuration file.
    Args:
        config (Dict): experiment config.
        send_notification_slack (bool): If True, send notification to Slack.
    """
    try:
        exp_name = config.get("exp_name")

        setup_logging(exp_name, config)

        start_time = time.time()

        all_test_results = []

        if config.get("method", {}).get("type") == "unsupervised":
            logging.info("Running unsupervised method.")
            unsupervised(config, data_path)
        else:
            logging.info("Running supervised method.")
            all_test_results = supervised(config, data_path)

        end_time = time.time()
        logging.info(f"Experiment {exp_name} finished in {end_time - start_time:.2f} seconds.")
        if send_notification_slack:
            client = setup_slack()
            if all_test_results: # Check if there are results to format
                slack_msg_blocks = format_results_to_slack_blocks(all_test_results[0][2])  # TODO: Handle multiple tasks if needed  # BUG: error data format due to attr updates
                # Use backticks for experiment name for better visibility
                message = f"✅ Experiment `{exp_name}` finished successfully. Here are the results."
            else: # Handle cases with no specific test results (e.g., unsupervised run finished)
                message = f"✅ Experiment `{exp_name}` finished successfully. (No specific test results to display)."
                slack_msg_blocks = None
            send_slack_message(client, "#training-notifications", message, blocks=slack_msg_blocks)
            

    except Exception as e:
        logging.error(f"Error running experiment with config {config['_config_path_']}: {e}", exc_info=True)
        if send_notification_slack:
            client = setup_slack()
            send_slack_message(client, "#training-notifications", f"❌ Experiment {exp_name} failed with error: {e}")


if __name__ == '__main__':
    # Default single config path (can be overridden or ignored)
    default_config_path = "config/supervised/ring1/hr/ir/resnet-ring1-hr-all-ir.json"

    warnings.filterwarnings('ignore', category=UserWarning, module='torch.nn')

    parser = argparse.ArgumentParser(description='RingTool.')
    parser.add_argument('--data-path', type=str, default=None, help='Path to the data folder.')
    parser.add_argument('--send-notification-slack', action="store_true", help='Send notification to slack.')
    parser.add_argument('--save-predictions', action="store_true", help='Save detailed prediction pairs to predictions directory.')

    # --- Group for mutually exclusive config options ---
    group = parser.add_mutually_exclusive_group(required=False) # Make the group itself not strictly required initially

    group.add_argument('--config', type=str, default=None, help=f'Path to a single configuration JSON file (default if no batch dirs: {default_config_path}).')
    group.add_argument('--batch-configs-dirs', type=str, nargs='+', help='One or more paths to directories containing configuration JSON files. Executes all found JSONs.')
    # --- End of group ---

    args = parser.parse_args()

    data_path = args.data_path
    if not data_path or not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path {data_path} does not exist.")
    if not os.path.isdir(data_path):
        raise NotADirectoryError(f"Data path {data_path} is not a directory.")

    batch_configs_dirs = args.batch_configs_dirs # This will be a list of paths or None
    single_config_path = args.config
    send_notification_slack = args.send_notification_slack
    save_predictions = args.save_predictions

    config_files_to_run = []

    if batch_configs_dirs:
        logging.info(f"Scanning for JSON config files in directories: {', '.join(batch_configs_dirs)}")
        for config_dir in batch_configs_dirs:
            if not os.path.isdir(config_dir):
                logging.warning(f"Provided batch config path is not a directory, skipping: {config_dir}")
                continue
            
            found_in_dir = 0
            for root, _, files in os.walk(config_dir):
                for file in files:
                    if file.endswith(".json"):
                        full_path = os.path.join(root, file)
                        config_files_to_run.append(full_path)
                        found_in_dir += 1
            logging.info(f"Found {found_in_dir} JSON files in {config_dir}")

        if not config_files_to_run:
             logging.warning("No JSON configuration files found in the specified batch directories.")
        
    elif single_config_path:
        if os.path.isfile(single_config_path):
             config_files_to_run.append(single_config_path)
        else:
             logging.error(f"Specified single config file not found: {single_config_path}")
             exit(1)

    elif not batch_configs_dirs and not single_config_path:
        # Neither batch dirs nor a single config was specified, try the default
        logging.info(f"Neither --config nor --batch-configs-dirs specified. Trying default config: {default_config_path}")
        if os.path.isfile(default_config_path):
            config_files_to_run.append(default_config_path)
            single_config_path = default_config_path # Update for logging clarity later
        else:
            logging.error(f"Default configuration file not found: {default_config_path}")
            parser.print_help()
            exit(1)


    # --- Run Experiments ---
    if config_files_to_run:
        logging.info(f"Found {len(config_files_to_run)} configuration file(s) to process.")
        
        total_configs = len(config_files_to_run)
        for i, config_file_path in enumerate(config_files_to_run, 1):
            logging.info(f"--- Running experiment {i}/{total_configs} with config: {config_file_path} ---")
            try:
                config = load_config(config_file_path)
                if config is None:
                     logging.warning(f"Skipping experiment due to load failure for {config_file_path}")
                     continue # Skip to the next config file
                
                # Add config path to config dict for potential logging inside do_run_experiment
                config['_config_path_'] = config_file_path 
                config['_save_predictions_'] = save_predictions  # Pass flag to experiment
                
                do_run_experiment(config, data_path, send_notification_slack)
                
            except Exception as e:
                logging.error(f"!!! Critical error during experiment with config {config_file_path}: {e}", exc_info=True)

        logging.info("--- Finished processing all specified configurations. ---")
    else:
        logging.error("No valid configuration files found or specified to run.")
        if not batch_configs_dirs and not single_config_path: # If user provided nothing
            parser.print_help()
