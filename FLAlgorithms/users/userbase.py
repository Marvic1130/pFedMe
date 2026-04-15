import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
import numpy as np
import copy


def _is_cuda_device(device) -> bool:
    try:
        if isinstance(device, torch.device):
            return device.type == "cuda"
        return str(device).startswith("cuda")
    except Exception:
        return False

class User:
    """
    Base class for users in federated learning.
    """
    def __init__(self, device, id, train_data, test_data, model, batch_size = 0, learning_rate = 0, beta = 0 , lamda = 0, local_epochs = 0):

        self.device = device
        self.model = copy.deepcopy(model)
        self.id = id  # integer
        self.train_samples = len(train_data)
        self.test_samples = len(test_data)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta = beta
        self.lamda = lamda
        self.local_epochs = local_epochs

        self._pin_memory = bool(_is_cuda_device(self.device))
        self.trainloader = DataLoader(
            train_data,
            self.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=self._pin_memory,
        )
        self.testloader = DataLoader(
            test_data,
            self.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=self._pin_memory,
        )
        # Ensure a positive batch size for the full test loader.
        # Some clients may have zero test samples when using auto-selection/holdout routing.
        # Torch DataLoader requires batch_size >= 1; using 1 yields an empty iterator for zero-length datasets.
        full_test_bs = max(1, self.test_samples)
        self.testloaderfull = DataLoader(
            test_data, batch_size=full_test_bs, pin_memory=self._pin_memory
        )
        self.trainloaderfull = DataLoader(
            train_data, self.train_samples, pin_memory=self._pin_memory
        )
        self.iter_trainloader = iter(self.trainloader)
        self.iter_testloader = iter(self.testloader)

        # those parameters are for persionalized federated learing.
        self.local_model = copy.deepcopy(list(self.model.parameters()))
        self.persionalized_model = copy.deepcopy(list(self.model.parameters()))
        self.persionalized_model_bar = copy.deepcopy(list(self.model.parameters()))
    
    def set_parameters(self, model):
        for old_param, new_param, local_param in zip(self.model.parameters(), model.parameters(), self.local_model):
            old_param.data = new_param.data.clone()
            local_param.data = new_param.data.clone()
        #self.local_weight_updated = copy.deepcopy(self.optimizer.param_groups[0]['params'])

    def get_parameters(self):
        for param in self.model.parameters():
            param.detach()
        return self.model.parameters()
    
    def clone_model_paramenter(self, param, clone_param):
        for param, clone_param in zip(param, clone_param):
            clone_param.data = param.data.clone()
        return clone_param
    
    def get_updated_parameters(self):
        return self.local_weight_updated
    
    def update_parameters(self, new_params):
        for param , new_param in zip(self.model.parameters(), new_params):
            param.data = new_param.data.clone()

    def get_grads(self):
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad.data)
        return grads

    def test(self):
        self.model.eval()
        non_blocking = _is_cuda_device(self.device)
        test_acc = 0
        loss_sum = 0.0
        total_samples = 0
        for x, y in self.testloaderfull:
            x = x.to(self.device, non_blocking=non_blocking)
            y = y.to(self.device, non_blocking=non_blocking)
            output = self.model(x)
            batch_size = y.shape[0]
            total_samples += batch_size
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            loss_sum += self.loss(output, y).item() * batch_size
        if total_samples == 0:
            return 0, 0.0, 0
        avg_loss = loss_sum / total_samples
        return test_acc, avg_loss, total_samples

    def train_error_and_loss(self):
        self.model.eval()
        non_blocking = _is_cuda_device(self.device)
        train_acc = 0
        loss = 0
        for x, y in self.trainloaderfull:
            x = x.to(self.device, non_blocking=non_blocking)
            y = y.to(self.device, non_blocking=non_blocking)
            output = self.model(x)
            train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            loss += self.loss(output, y)
            #print(self.id + ", Train Accuracy:", train_acc)
            #print(self.id + ", Train Loss:", loss)
        return train_acc, loss , self.train_samples
    
    def test_persionalized_model(self):
        self.model.eval()
        non_blocking = _is_cuda_device(self.device)
        test_acc = 0
        loss_sum = 0.0
        total_samples = 0
        self.update_parameters(self.persionalized_model_bar)
        for x, y in self.testloaderfull:
            x = x.to(self.device, non_blocking=non_blocking)
            y = y.to(self.device, non_blocking=non_blocking)
            output = self.model(x)
            batch_size = y.shape[0]
            total_samples += batch_size
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            loss_sum += self.loss(output, y).item() * batch_size
        self.update_parameters(self.local_model)
        if total_samples == 0:
            return 0, 0.0, 0
        avg_loss = loss_sum / total_samples
        return test_acc, avg_loss, total_samples

    def train_error_and_loss_persionalized_model(self):
        self.model.eval()
        non_blocking = _is_cuda_device(self.device)
        train_acc = 0
        loss = 0
        self.update_parameters(self.persionalized_model_bar)
        for x, y in self.trainloaderfull:
            x = x.to(self.device, non_blocking=non_blocking)
            y = y.to(self.device, non_blocking=non_blocking)
            output = self.model(x)
            train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            loss += self.loss(output, y)
            #print(self.id + ", Train Accuracy:", train_acc)
            #print(self.id + ", Train Loss:", loss)
        self.update_parameters(self.local_model)
        return train_acc, loss , self.train_samples
    
    def get_next_train_batch(self):
        try:
            # Samples a new batch for persionalizing
            (X, y) = next(self.iter_trainloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_trainloader = iter(self.trainloader)
            try:
                (X, y) = next(self.iter_trainloader)
            except StopIteration:
                # Loader is empty (likely due to drop_last=True with small dataset)
                dataset_len = len(self.trainloader.dataset)
                if dataset_len > 1:
                    # Fallback: use dataset without drop_last (batch size will be dataset_len)
                    # This is safe for BatchNorm since dataset_len > 1
                    temp_loader = DataLoader(
                        self.trainloader.dataset,
                        batch_size=self.batch_size,
                        shuffle=True,
                        drop_last=False,
                        pin_memory=self._pin_memory,
                    )
                    (X, y) = next(iter(temp_loader))
                elif dataset_len == 1:
                    # Single sample case. BatchNorm will fail with batch size 1.
                    # Duplicate the sample to make batch size 2.
                    temp_loader = DataLoader(
                        self.trainloader.dataset,
                        batch_size=1,
                        shuffle=True,
                        drop_last=False,
                        pin_memory=self._pin_memory,
                    )
                    (X, y) = next(iter(temp_loader))
                    X = torch.cat([X, X], dim=0)
                    y = torch.cat([y, y], dim=0)
                else:
                    raise RuntimeError(f"Client {self.id} has no training data!")
        non_blocking = _is_cuda_device(self.device)
        return (
            X.to(self.device, non_blocking=non_blocking),
            y.to(self.device, non_blocking=non_blocking),
        )
    
    def get_next_test_batch(self):
        try:
            # Samples a new batch for persionalizing
            (X, y) = next(self.iter_testloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_testloader = iter(self.testloader)
            try:
                (X, y) = next(self.iter_testloader)
            except StopIteration:
                # If still empty (e.g. dataset too small for drop_last=True), use full loader or handle gracefully
                # For now, try to get from full loader or raise error if empty
                if len(self.testloader) == 0 and len(self.testloader.dataset) > 0:
                     # Fallback: create a loader without drop_last just for this batch if possible, 
                     # or just return the whole dataset if it fits in memory (which is what testloaderfull does)
                     # But we need a batch.
                     # Let's try to get from trainloader as fallback? No, that's wrong.
                     # If drop_last killed all data, we must allow a smaller batch or duplicate data.
                     # Let's try to fetch from testloaderfull but take only batch_size
                     loader = DataLoader(
                         self.testloader.dataset,
                         batch_size=self.batch_size,
                         shuffle=True,
                         drop_last=False,
                         pin_memory=self._pin_memory,
                     )
                     self.iter_testloader = iter(loader)
                     (X, y) = next(self.iter_testloader)
                else:
                     raise
        non_blocking = _is_cuda_device(self.device)
        return (
            X.to(self.device, non_blocking=non_blocking),
            y.to(self.device, non_blocking=non_blocking),
        )

    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "user_" + self.id + ".pt"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        self.model = torch.load(os.path.join(model_path, "server" + ".pt"))
    
    @staticmethod
    def model_exists():
        return os.path.exists(os.path.join("models", "server" + ".pt"))