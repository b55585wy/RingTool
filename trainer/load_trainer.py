import logging
import os
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from nets.load_model import MODEL_CLASSES, SupportedSupervisedModels
from unsupervised.hr.hr import get_hr
from unsupervised.rr.rr import get_rr
from unsupervised.spo2.spo2 import get_spo2
from utils.utils import calculate_metrics, plot_and_save_metrics, save_config, save_metrics_to_csv
from utils.utils import physiological_filter


class BaseTrainer:
    def __init__(self, model, config: Dict):
        self.config = config
        if config["method"]["type"] == "ML":
            model_name = SupportedSupervisedModels(config["method"]["name"].lower())  # Convert to enum
        
            if model_name in MODEL_CLASSES:
                self.model = model
                self.device = torch.device(
                    "cuda:" + str(config["train"]["device"])
                    if torch.cuda.is_available() and config["train"]["device"] != "cpu"
                    else "cpu"
                )
                self.model.to(self.device)
            else:
                raise ValueError(f"Unsupported model: {config['method']['name']}")

    def load_optimizer(self):
        raise NotImplementedError("Should implement load_optimizer method in subclass")

    def load_criterion(self):
        raise NotImplementedError("Should implement load_criterion method in subclass")

    def fit(self, train_loader, valid_loader, task=None, fold=None):
        raise NotImplementedError("Subclass must implement fit method")

    def test(self, test_loader, checkpoint_path=None, task=None):
        raise NotImplementedError("Subclass must implement test method")


class UnsupervisedTester(BaseTrainer):
    def __init__(self, model, config: Dict):
        super().__init__(model, config)
        
    def load_optimizer(self):
        # Not needed for unsupervised testing
        pass

    def load_criterion(self):
        # Not needed for unsupervised testing
        pass

    def fit(self, train_loader, valid_loader, task=None, fold=None):
        # Not used in unsupervised approach
        logging.info("Unsupervised methods do not require fitting/training")
        return None

    def test(self, test_loader, checkpoint_path=None, task="hr"):
        if self.config["method"]["name"] not in ["peak", "fft", "ratio"]:
            raise ValueError("This tester is only for unsupervised methods, choose from 'peak', 'fft', 'ratio'")

        logging.info(f"Running unsupervised testing for {task}...")
        all_predictions = []
        all_targets = []
        
        algorithm = self.config["method"].get("name", "peak")
        logging.info(f"Using algorithm: {algorithm} for task: {task}")

        for inputs, labels in tqdm(test_loader, desc=f"Testing {task}"):
            # Convert tensors to numpy for processing
            inputs_np = inputs.cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            batch_size = inputs_np.shape[0]
            batch_predictions = []
            
            # Process each sample in the batch
            for i in range(batch_size):
                signal = inputs_np[i]
                
                # Call the appropriate unsupervised method based on task
                if task in ["hr", "bvp_hr", "samsung_hr", "oura_hr"]:
                    prediction = get_hr(signal, method=algorithm)
                elif task == "resp_rr":
                    prediction = get_rr(signal, method=algorithm)
                elif task == "spo2":
                    ppg_ir = signal[:, 0]
                    ppg_red = signal[:, 1]
                    prediction = get_spo2(ppg_ir, ppg_red, ring_type=self.config["dataset"].get("ring_type", "ring1"),method=algorithm)
                else:
                    raise ValueError(f"Unsupported task: {task}")
                
                batch_predictions.append(prediction)
            
            all_predictions.extend(batch_predictions)
            all_targets.extend(labels_np.reshape(-1).tolist())

        # Convert to tensors for metrics calculation
        all_predictions = torch.tensor(all_predictions).reshape(-1, 1)
        all_targets = torch.tensor(all_targets).reshape(-1, 1)
        
        # Apply physiological range filtering
        try:
            filter_task = task
            for _suffix in ("_stationary", "_motion", "_spo2"):
                if filter_task.endswith(_suffix):
                    filter_task = filter_task[: -len(_suffix)]
                    break
            mask_u = physiological_filter(all_targets, filter_task, behavior="mask")
            if not torch.is_tensor(mask_u):
                mask_u = torch.from_numpy(mask_u)
            mask_u = mask_u.to(dtype=torch.bool)
            all_predictions = all_predictions[mask_u]
            all_targets = all_targets[mask_u]
        except Exception as e:
            logging.warning(f"physiological_filter failed in UnsupervisedTester.test for task {task}: {e}")
        
        # Calculate metrics
        metrics = calculate_metrics(all_predictions, all_targets)
        logging.debug(f"Task: {task} - "
              f"MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}, "
              f"MAPE: {metrics['mape']:.2f}%, Pearson: {metrics['pearson']:.4f}")
        
        # Save metrics to CSV
        save_metrics_to_csv(metrics, self.config, task)
        
        # Plot and save metrics
        plot_and_save_metrics(predictions=all_predictions, targets=all_targets, config=self.config, task=task)
        
        return {
            "loss": 0,  # No loss computation in unsupervised methods
            **metrics
        }


class SupervisedTrainer(BaseTrainer):
    def __init__(self, model, config: Dict, eval_func=None):
        super().__init__(model, config)
        self.eval_func = eval_func
        self.load_optimizer()
        self.load_criterion()
        self.gradient_accum = config["train"].get("gradient_accum", 1)

    def load_criterion(self):
        criterion_type = self.config["train"]["criterion"]
        if criterion_type == "cross_entropy":
            self.criterion = nn.CrossEntropyLoss()
        elif criterion_type == "mse":
            self.criterion = nn.MSELoss()
        elif criterion_type == "mae":
            self.criterion = lambda x,y:(x-y).abs().mean()
        else:
            raise ValueError(f"Unsupported criterion type: {criterion_type}")

    def load_optimizer(self):
        optimizer_type = self.config["train"]["optimizer"]
        if optimizer_type == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.get("lr", 0.001))
        elif optimizer_type == "adamw":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.get("lr", 0.001))
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    def fit(self, train_loader, valid_loader, task=None, fold=None) -> Tuple[str, str]:
        scaler = GradScaler(enabled=True)
        epochs = self.config.get("train", {}).get("epochs", 200)
        best_loss = float('inf')  # For metrics like loss where lower is better
        
        # Early stopping setup
        early_stopping = self.config.get("train", {}).get("early_stopping", {})
        early_stop = False
        early_stopping_patience = early_stopping.get("patience", 40)  # Default patience is 40 epochs
        monitor = early_stopping.get("monitor", "val_loss")  # Metric to monitor
        mode = early_stopping.get("mode", "min")  # 'min' for loss, 'max' for metrics like accuracy
        counter = 0  # Counter for early stopping
        best_score = float('inf') if mode == "min" else float('-inf')

        scheduler = None
        if self.config.get("train", {}).get("scheduler", {}).get("type", None) == "reduce_on_plateau":
            factor = self.config.get("train", {}).get("scheduler", {}).get("factor", 0.5)
            patience = self.config.get("train", {}).get("scheduler", {}).get("patience", 10)
            min_lr = self.config.get("train", {}).get("scheduler", {}).get("min_lr", 1e-6)
            threshold = self.config.get("train", {}).get("scheduler", {}).get("threshold", 1e-4)
            # Some torch versions do not support `verbose` in ReduceLROnPlateau
            try:
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode='min',
                    factor=factor,
                    patience=patience,
                    threshold=threshold,
                    cooldown=0,
                    min_lr=min_lr,
                    verbose=True,
                )
            except TypeError:
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode='min',
                    factor=factor,
                    patience=patience,
                    threshold=threshold,
                    cooldown=0,
                    min_lr=min_lr,
                )
        exp_name = self.config.get("exp_name", "default_experiment")
        checkpoint_dir = os.path.join("models", exp_name, task, fold)
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_checkpoint_path = os.path.join(checkpoint_dir, f"{exp_name}_{task}_{fold}_best.pt")
        config_save_path = os.path.join(checkpoint_dir, f"{exp_name}.json")
        if self.gradient_accum > 1:
            logging.info(f"Training with gradient accumulation: {self.gradient_accum} steps")
        
        # Log early stopping config if enabled
        if early_stopping:
            logging.info(f"Early stopping enabled with patience={early_stopping_patience}, monitoring={monitor}, mode={mode}")
            
        progress_bar = tqdm(range(epochs), desc="Training Progress")
        for epoch in progress_bar:
            self.model.train()
            train_loss = 0
            self.optimizer.zero_grad()
            for idx, (inputs, labels) in enumerate(train_loader):
                # Gradient accumulation

                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Check for NaNs or large values in inputs
                if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                    logging.error(f"NaN or Inf found in inputs at epoch {epoch+1}, batch {idx}. Skipping batch.")
                    continue
                if inputs.abs().max() > 1e6: # Check for abnormally large values (threshold can be adjusted)
                     logging.warning(f"Input values might be too large at epoch {epoch+1}, batch {idx}. Max value: {inputs.abs().max()}")


                with autocast():
                    outputs, _ = self.model(inputs)
                    # Filter out samples with labels outside physiological range before loss computation
                    try:
                        filter_task = task
                        for _suffix in ("_stationary", "_motion", "_spo2"):
                            if filter_task.endswith(_suffix):
                                filter_task = filter_task[: -len(_suffix)]
                                break
                        mask_np = physiological_filter(labels, filter_task, behavior="mask")
                        if not torch.is_tensor(mask_np):
                            mask = torch.from_numpy(mask_np).to(device=self.device, dtype=torch.bool)
                        else:
                            mask = mask_np.to(device=self.device, dtype=torch.bool)
                        if mask.ndim > 1:
                            mask = mask.squeeze(-1)
                        if mask.sum().item() == 0:
                            raise ValueError("All samples masked in this batch; skipping batch.")
                        outputs_m = outputs[mask]
                        labels_m = labels[mask]
                    except Exception as e:
                        # Use unfiltered data if physiological filtering fails
                        logging.warning(f"physiological mask failed in train batch {idx}: {e}")
                        outputs_m, labels_m = outputs, labels

                    loss = self.criterion(outputs_m, labels_m)

                # Check for NaNs in loss
                if torch.isnan(loss):
                    logging.error(f"NaN loss detected at epoch {epoch+1}, batch {idx}. Skipping optimizer step.")
                    # Optionally: investigate inputs/outputs/labels that caused NaN
                    # logging.error(f"Inputs max: {inputs.max()}, min: {inputs.min()}")
                    #logging.error(f"Outputs: {outputs}")
                    #logging.error(f"Labels: {labels}")
                    continue # Skip backpropagation for this batch if loss is NaN

                scaled_loss = scaler.scale(loss)
                scaled_loss = scaled_loss / self.gradient_accum
                scaled_loss.backward()

                if (idx + 1) % self.gradient_accum == 0:
                    # Add gradient clipping before the optimizer step
                    scaler.unscale_(self.optimizer) # Unscale gradients before clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0) # Clip gradients (max_norm=1.0 is a common starting point)

                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad()
                train_loss += loss.item() / self.gradient_accum

            if len(train_loader) % self.gradient_accum != 0:
                # Apply gradient clipping here as well if the last step wasn't taken inside the loop
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                scaler.step(self.optimizer)
                scaler.update() # scaler.update() should be called once per iteration, typically after optimizer.step()
                self.optimizer.zero_grad()
            
            # Validation
            self.model.eval()
            valid_loss = 0
            all_preds, all_targets = [], []
            with torch.no_grad():
                for inputs, labels in valid_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs, _ = self.model(inputs)
                    # Filter validation samples outside physiological range
                    try:
                        filter_task = task
                        for _suffix in ("_stationary", "_motion", "_spo2"):
                            if filter_task.endswith(_suffix):
                                filter_task = filter_task[: -len(_suffix)]
                                break
                        mask_np = physiological_filter(labels, filter_task, behavior="mask")
                        if not torch.is_tensor(mask_np):
                            mask = torch.from_numpy(mask_np).to(device=self.device, dtype=torch.bool)
                        else:
                            mask = mask_np.to(device=self.device, dtype=torch.bool)
                        if mask.ndim > 1:
                            mask = mask.squeeze(-1)
                        if mask.sum().item() == 0:
                            # Skip batch when all samples are filtered out
                            continue
                        outputs_m = outputs[mask]
                        labels_m = labels[mask]
                    except Exception as e:
                        logging.warning(f"physiological mask failed in valid batch: {e}")
                        outputs_m, labels_m = outputs, labels

                    loss = self.criterion(outputs_m, labels_m)
                    valid_loss += loss.item()
                    all_preds.append(outputs_m.cpu())
                    all_targets.append(labels_m.cpu())
            
            all_preds = torch.cat(all_preds, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            
            # Apply physiological range filtering for validation
            try:
                filter_task = task
                for _suffix in ("_stationary", "_motion", "_spo2"):
                    if filter_task.endswith(_suffix):
                        filter_task = filter_task[: -len(_suffix)]
                        break
                mask_val = physiological_filter(all_targets, filter_task, behavior="mask")
                if not torch.is_tensor(mask_val):
                    mask_val = torch.from_numpy(mask_val)
                mask_val = mask_val.to(dtype=torch.bool)
                all_preds = all_preds[mask_val]
                all_targets = all_targets[mask_val]
            except Exception as e:
                logging.warning(f"physiological_filter failed in validation for task {task}: {e}")
            
            # Update progress bar with metrics
            train_loss_avg = train_loss / len(train_loader)
            valid_loss_avg = valid_loss / len(valid_loader)
            metrics = calculate_metrics(all_preds, all_targets)
            
            # Get current score for early stopping
            current_score = valid_loss_avg if monitor == "val_loss" else metrics.get(monitor, valid_loss_avg)
            
            # Save checkpoint if current model is better
            if valid_loss_avg < best_loss:
                best_loss = valid_loss_avg
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'valid_loss': valid_loss_avg,
                    'metrics': metrics
                }, best_checkpoint_path)
                # message = f"Epoch {epoch+1}: Saved best model with validation loss={valid_loss_avg:.4f}"
                # logging.info(message)
                progress_bar.set_description(f"Training Progress (saved best model, val_loss={valid_loss_avg:.4f})")
            
            # Check early stopping conditions
            if early_stopping:
                improved = (mode == "min" and current_score < best_score) or (mode == "max" and current_score > best_score)
                if improved:
                    best_score = current_score
                    counter = 0
                else:
                    counter += 1
                    if counter >= early_stopping_patience:
                        logging.info(f"Early stopping triggered after {epoch+1} epochs! No improvement in {monitor} for {early_stopping_patience} epochs.")
                        early_stop = True

            if scheduler:
                scheduler.step(valid_loss_avg)

            progress_bar.set_postfix(
                epoch=f"{epoch+1}/{epochs}",
                task=task,
                train_loss=f"{train_loss_avg:.4f}",
                val_loss=f"{valid_loss_avg:.4f}",
                mae=f"{metrics['mae']:.4f}",
                learning_rate=f"{self.optimizer.param_groups[0]['lr']:.8f}",
                early_stop_count=f"{counter}/{early_stopping_patience}" if early_stopping else "N/A"
            )

            # Print detailed metrics every 10 epochs
            if epoch % 10 == 0 or epoch == epochs-1:
                logging.info(f"\nEpoch {epoch+1}/{epochs}:  Task: {task}")
                logging.debug(f"  Training Loss: {train_loss_avg:.4f}")
                logging.debug(f"  Validation Loss: {valid_loss_avg:.4f}, "
                    f"MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}, "
                    f"MAPE: {metrics['mape']:.2f}%, Pearson: {metrics['pearson']:.4f}")
                
                if self.eval_func is not None:
                    score = self.eval_func(all_preds, all_targets)
                    logging.debug(f"Custom evaluation score: {score}")
            
            # Break the loop if early stopping is triggered
            if early_stop:
                logging.info("Training stopped early due to early stopping criteria.")
                break
        
        # save the model config file alongside the best model pt file
        if save_config(self.config, config_save_path):
            logging.info(f"Configuration saved to {config_save_path}")
        else:
            logging.error(f"Failed to save configuration to {config_save_path}")
            config_save_path = None

        return best_checkpoint_path, config_save_path

    def test(self, test_loader: DataLoader, checkpoint_path: str, task: str):
        # Load the best checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            logging.info(f"Loading best model checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            total_params = sum(p.numel() for p in self.model.state_dict().values())
            logging.debug(f"Model parameters: {total_params}")
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logging.debug(f"Loaded model from epoch {checkpoint['epoch']+1} with "
                  f"validation loss: {checkpoint['valid_loss']:.4f}, "
                  f"MAE: {checkpoint['metrics']['mae']:.4f}")
        
        self.model.eval()
        test_loss = 0
        all_preds, all_targets = [], []
        all_metadata = []  # Collect metadata for each sample
        with torch.no_grad():
            for batch_data in tqdm(test_loader, desc="Testing"):
                # Handle both old format (inputs, labels) and new format (inputs, labels, metadata)
                if len(batch_data) == 3:
                    inputs, labels, metadata_list = batch_data
                    all_metadata.extend(metadata_list)
                else:
                    inputs, labels = batch_data
                    
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs, _ = self.model(inputs)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()
                all_preds.append(outputs.cpu())
                all_targets.append(labels.cpu())
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Apply physiological range filtering for test
        try:
            filter_task = task
            for _suffix in ("_stationary", "_motion", "_spo2"):
                if filter_task.endswith(_suffix):
                    filter_task = filter_task[: -len(_suffix)]
                    break
            mask_test = physiological_filter(all_targets, filter_task, behavior="mask")
            if not torch.is_tensor(mask_test):
                mask_test = torch.from_numpy(mask_test)
            mask_test = mask_test.to(dtype=torch.bool)
            all_preds = all_preds[mask_test]
            all_targets = all_targets[mask_test]
        except Exception as e:
            logging.warning(f"physiological_filter failed in SupervisedTrainer.test for task {task}: {e}")
        
        metrics = calculate_metrics(all_preds, all_targets)
        logging.critical(f"Task:{task} Test Loss: {test_loss / len(test_loader):.4f}, "
              f"MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}, "
              f"MAPE: {metrics['mape']:.2f}%, Pearson: {metrics['pearson']:.4f}")
        
        # Save metrics to CSV
        _ = save_metrics_to_csv(metrics, self.config, task)
        # Plot and save metrics
        _ = plot_and_save_metrics(predictions=all_preds, targets=all_targets, config=self.config, task=task)
       
        if self.eval_func is not None:
            score = self.eval_func(all_preds, all_targets)
            logging.debug(f"Custom evaluation score: {score}")

        return {
            "preds_and_targets": (all_preds, all_targets),
            "metadata": all_metadata if all_metadata else None,  # Include metadata if available
            "loss": test_loss / len(test_loader),
            **metrics
        }


def load_trainer(model, model_name: str, config: Dict):
    """
    Load the appropriate trainer based on the model name and configuration.
    """
    # Case 1: Unsupervised models
    if model_name in ["peak", "fft", "ratio"]:
        return UnsupervisedTester(model, config)
    # Case 2: Supervised models
    try:
        # Convert model name to enum
        _ = SupportedSupervisedModels(model_name.lower())
        return SupervisedTrainer(model, config)
    except ValueError:
        # If the model name is not in the enum, it might be a custom model
        logging.error(f"Model name '{model_name}' not found in SupportedSupervisedModels.")
    
    # Default case: return a base trainer
    # This is a fallback and should not be reached if the above cases are handled correctly
    # You can also add a warning or error log here if needed
    logging.warning(f"Using BaseTrainer for model: {model_name}")
    return BaseTrainer(model, config)
