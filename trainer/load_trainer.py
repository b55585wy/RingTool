from typing import Dict
import torch
import torch.nn as nn
from tqdm import tqdm

# -------------------------------
# 工具函数：指标计算
# -------------------------------

def calculate_mae(predictions, targets):
    """计算平均绝对误差 (MAE)"""
    return torch.mean(torch.abs(predictions - targets)).item()

def calculate_rmse(predictions, targets):
    """计算均方根误差 (RMSE)"""
    return torch.sqrt(torch.mean((predictions - targets) ** 2)).item()

def calculate_mape(predictions, targets):
    """计算平均绝对百分比误差 (MAPE)"""
    mask = targets != 0
    if mask.sum() > 0:
        mape = torch.mean(torch.abs((targets[mask] - predictions[mask]) / targets[mask])) * 100
        return mape.item() if isinstance(mape, torch.Tensor) else mape
    return float('inf')

def calculate_pearson(predictions, targets):
    """计算皮尔逊相关系数"""
    x = predictions.flatten()
    y = targets.flatten()
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    pearson = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    return pearson.item()

def calculate_metrics(predictions, targets):
    """计算所有指标"""
    return {
        "mae": calculate_mae(predictions, targets),
        "rmse": calculate_rmse(predictions, targets),
        "mape": calculate_mape(predictions, targets),
        "pearson": calculate_pearson(predictions, targets)
    }

# -------------------------------
# 基础训练器
# -------------------------------

class BaseTrainer:
    def __init__(self, model, config: Dict):
        self.config = config
        self.model = model
        self.device = torch.device("cuda:" + str(config["device"]) 
                                   if torch.cuda.is_available() and config["device"] != "cpu" 
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
        criterion_type = self.config.get("criterion")
        if criterion_type == "cross entropy":
            self.criterion = nn.CrossEntropyLoss()
        elif criterion_type == "mse":
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported criterion type: {criterion_type}")

    def load_optimizer(self):
        optimizer_type = self.config.get("optimizer")
        if optimizer_type == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.get("lr", 0.001))
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    def fit(self, train_loader, valid_loader):
        epochs = self.config.get("epochs", 1)
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0
            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs, _ = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            print(f"Training Loss: {train_loss / len(train_loader):.4f}")

            # 验证阶段
            self.model.eval()
            valid_loss = 0
            all_preds, all_targets = [], []
            with torch.no_grad():
                for inputs, labels in tqdm(valid_loader, desc="Validating"):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs, _ = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    valid_loss += loss.item()
                    all_preds.append(outputs.cpu())
                    all_targets.append(labels.cpu())
            
            all_preds = torch.cat(all_preds, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            metrics = calculate_metrics(all_preds, all_targets)
            print(f"Validation Loss: {valid_loss / len(valid_loader):.4f}, "
                  f"MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}, "
                  f"MAPE: {metrics['mape']:.2f}%, Pearson: {metrics['pearson']:.4f}")

            if self.eval_func is not None:
                score = self.eval_func(all_preds, all_targets)
                print(f"Custom evaluation score: {score}")

    def test(self, test_loader):
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
