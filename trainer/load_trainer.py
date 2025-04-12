from typing import Dict
import torch
import torch.nn as nn
from tqdm import tqdm
import os
import numpy as np
from scipy.stats import gaussian_kde
import pandas as pd
# from utils import *
from utils.utils import calculate_metrics, plot_metrics

class BaseTrainer:
    def __init__(self, model, config: Dict):
        self.config = config
        self.model = model
        self.device = torch.device("cuda:" + str(config["train"]["device"]) 
                                   if torch.cuda.is_available() and config["train"]["device"] != "cpu" 
                                   else "cpu")
        self.model.to(self.device)


    def load_optimizer(self):
        """加载优化器"""
        raise NotImplementedError("子类需要实现 load_optimizer 方法")

    def load_criterion(self):
        """加载损失函数"""
        raise NotImplementedError("子类需要实现 load_criterion 方法")

    def fit(self, train_loader, valid_loader):
        """训练模型"""
        raise NotImplementedError("子类需要实现 fit 方法")

    def test(self, test_loader):
        """测试模型"""
        raise NotImplementedError("子类需要实现 test 方法")

# -------------------------------
# 监督训练器
# -------------------------------

class SupervisedTrainer(BaseTrainer):
    def __init__(self, model, config: Dict, eval_func=None):
        super().__init__(model, config)
        self.eval_func = eval_func
        self.load_optimizer()
        self.load_criterion()

    def load_criterion(self):
        criterion_type = self.config["train"]["criterion"]
        if criterion_type == "cross entropy":
            self.criterion = nn.CrossEntropyLoss()
        elif criterion_type == "mse":
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported criterion type: {criterion_type}")

    def load_optimizer(self):
        optimizer_type = self.config["train"]["optimizer"]
        if optimizer_type == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.get("lr", 0.001))
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    def fit(self, train_loader, valid_loader,task):
        epochs = self.config.get("train", {}).get("epochs", 200)
        best_loss = float('inf')  # For metrics like loss where lower is better
        # patience = self.config.get("train", {}).get("early_stopping", {}).get("patience", 10)
        checkpoint_dir = os.path.join("models", self.config.get("exp_name"),task)
        model_name = self.config.get("exp_name")
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_best.pt")
        
        progress_bar = tqdm(range(epochs), desc="Training Progress")
        for epoch in progress_bar:
            # 训练阶段
            self.model.train()
            train_loss = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs, _ = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            
            # 验证阶段
            self.model.eval()
            valid_loss = 0
            all_preds, all_targets = [], []
            with torch.no_grad():
                for inputs, labels in valid_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs, _ = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    valid_loss += loss.item()
                    all_preds.append(outputs.cpu())
                    all_targets.append(labels.cpu())
            
            all_preds = torch.cat(all_preds, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            
            # Update progress bar with metrics
            train_loss_avg = train_loss / len(train_loader)
            valid_loss_avg = valid_loss / len(valid_loader)
            metrics = calculate_metrics(all_preds, all_targets)
            
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
                progress_bar.set_description(f"Training Progress (saved best model, val_loss={valid_loss_avg:.4f})")

            
            progress_bar.set_postfix(
                epoch=f"{epoch+1}/{epochs}",
                train_loss=f"{train_loss_avg:.4f}",
                val_loss=f"{valid_loss_avg:.4f}",
                mae=f"{metrics['mae']:.4f}"
            )
            
            # Print detailed metrics every 10 epochs
            if epoch % 10 == 0 or epoch == epochs-1:
                print(f"\nEpoch {epoch+1}/{epochs}:")
                print(f"  Training Loss: {train_loss_avg:.4f}")
                print(f"  Validation Loss: {valid_loss_avg:.4f}, "
                    f"MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}, "
                    f"MAPE: {metrics['mape']:.2f}%, Pearson: {metrics['pearson']:.4f}")
                
                if self.eval_func is not None:
                    score = self.eval_func(all_preds, all_targets)
                    print(f"  Custom evaluation score: {score}")
            

        
        return best_checkpoint_path

    def test(self, test_loader, checkpoint_path,task):
        # Load the best checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading best model checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from epoch {checkpoint['epoch']+1} with "
                  f"validation loss: {checkpoint['valid_loss']:.4f}, "
                  f"MAE: {checkpoint['metrics']['mae']:.4f}")
        
        self.model.eval()
        test_loss = 0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Testing"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs, _ = self.model(inputs)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()
                all_preds.append(outputs.cpu())
                all_targets.append(labels.cpu())
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        metrics = calculate_metrics(all_preds, all_targets)
        print(f"Test Loss: {test_loss / len(test_loader):.4f}, "
              f"MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}, "
              f"MAPE: {metrics['mape']:.2f}%, Pearson: {metrics['pearson']:.4f}")
        
        # save metrics and imgs
        exp_name = self.config.get("exp_name")
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Get RingTool directory
        
        result_csv_path = self.config.get("csv_path")
        if result_csv_path is None:
            csv_dir = os.path.join(base_dir, "csv", exp_name, task)
            os.makedirs(csv_dir, exist_ok=True)
            result_csv_path = os.path.join(csv_dir, f"{exp_name}.csv")
        
        img_path_folder = self.config.get("img_path")
        if img_path_folder is None:
            img_path_folder = os.path.join(base_dir, "img", exp_name)
        
        os.makedirs(os.path.dirname(result_csv_path), exist_ok=True)
        os.makedirs(img_path_folder, exist_ok=True)

        # Create a DataFrame with the metrics
        metrics_df = pd.DataFrame({
            'exp_name': [self.config.get("exp_name", "")],
            'ring_type': [self.config.get("dataset", {}).get("ring_type", "")],
            'task': [task],
            'input_type': [self.config.get("dataset", {}).get("input_type", "")],
            'dataset_task': [self.config.get("dataset", {}).get("task", "")],
            'method_name': [self.config.get("method", {}).get("name", "")],
            'epochs': [self.config.get("train", {}).get("epochs", "")],
            'lr': [self.config.get("train", {}).get("lr", "")],
            'criterion': [self.config.get("train", {}).get("criterion", "")],
            'mae_with_std': [metrics['mae_with_std']],
            'rmse_with_std': [metrics['rmse_with_std']],
            'mape_with_std': [metrics['mape_with_std']],
            'pearson_with_std': [metrics['pearson_with_std']]
        })

        # Check if file exists to determine if we need to write headers
        file_exists = os.path.isfile(result_csv_path)

        # Write to CSV (append if file exists, create new with headers if it doesn't)
        metrics_df.to_csv(result_csv_path, mode='a' if file_exists else 'w', 
                  header=not file_exists, index=False)
        
        
        plot_metrics(all_preds, all_targets, img_path_folder,task)
        print(f"Saved results to {result_csv_path}")
        print(f"Saved visualization plots to {img_path_folder}")

        if self.eval_func is not None:
            score = self.eval_func(all_preds, all_targets)
            print(f"Custom evaluation score: {score}")

        return {
            "loss": test_loss / len(test_loader),
            **metrics
        }

# -------------------------------
# 训练器选择加载
# -------------------------------

def load_trainer(model, model_name: str, config: Dict):
    """根据模型名称加载对应的训练器"""
    if model_name in ["resnet"]:
        return SupervisedTrainer(model, config)
    return BaseTrainer(model, config)

if __name__ == '__main__':
    import argparse
    import json

    def load_config(config_path):
        with open(config_path, 'r') as file:
            config = json.load(file)
        return config

    config = load_config("/home/disk2/disk/3/tjk/RingTool/config/Resnet.json")
    print(config)
    print(config.get("img_path"))
    
    