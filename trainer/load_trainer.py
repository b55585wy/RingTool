from typing import Dict
import torch
import torch.nn as nn
from tqdm import tqdm

def load_trainer(model, model_name: str, config: Dict):
    # load a trainer
    if model_name in ["resnet"]:
        return SupervisedTrainer(model, config)
    return BaseTrainer(model, config)


class BaseTrainer():
    def __init__(self, model, config: Dict):
        self.config = config
        self.model = model
        self.device = torch.device("cuda:" + str(config["device"]) if torch.cuda.is_available() and config["device"] != "cpu" else "cpu")
        self.model.to(self.device)
        
    def load_optimizer(self):
        # load an optimizer
        pass

    def load_criterion(self):
        # load a criterion
        pass
    
    def fit(self, train_loader, valid_loader):
        # fit the model
        pass
    
    def test(self, test_loader):
        # test the model
        pass
    
    
class SupervisedTrainer(BaseTrainer):
    def __init__(self, model, config: Dict, eval_func=None):
        super().__init__(model, config)
        self.load_optimizer()
        self.load_criterion()
        self.eval_func = eval_func
        
    def load_criterion(self):
        if self.config["criterion"] == "cross entropy":
            self.criterion = nn.CrossEntropyLoss()
        elif self.config["criterion"] == "mse":
            self.criterion = nn.MSELoss()
            
    def load_optimizer(self):
        if self.config["optimizer"] == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])
            
    def fit(self, train_loader, valid_loader):
        # fit the model
        # TODO: evaluation metrices 
        for epoch in range(self.config["epochs"]):
            self.model.train()
            for idx, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']}")):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs, _ = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            with torch.no_grad():
                self.model.eval()
                valid_loss = 0
                correct = 0
                total = 0
                for inputs, labels in tqdm(valid_loader, desc="Validating"):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs, _ = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    valid_loss += loss.item()
                    if self.eval_func is not None:
                        score = self.eval_func(outputs, labels)
                    
                    # _, predicted = torch.max(outputs.data, 1)
                    # total += labels.size(0)
                    # correct += (predicted == labels).sum().item()
                
                print(f"Validation Loss: {valid_loss/len(valid_loader)}, eval score: {score if self.eval_func is not None else 'N/A'}")