"""Wrappers for pFedMe/FedAvg/Per-FedAvg baselines."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, Optional, Union

from src.core import AppConfig
from src.client.models import get_model_class

from .FLAlgorithms.servers.serveravg import FedAvg as FedAvgServer
from .FLAlgorithms.servers.serverpFedMe import pFedMe as PFedMeServer
from .FLAlgorithms.servers.serverperavg import PerAvg as PerFedAvgServer
from .FLAlgorithms.trainmodel.models import DNN, FemnistNet, Mclr_Logistic, Net, CNNCifar
from .FLAlgorithms.users.userpFedMe import UserpFedMe
from .utils.plot_utils import average_data
from src.data.leaf_loader import load_leaf_splits_datasets, auto_select_leaf_splits
import torch


def _build_model(dataset: str, model_name: str, cfg: Optional[AppConfig] = None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ModelClass = get_model_class(dataset)
    if ModelClass:
        dname = str(dataset).lower()
        num_classes = int(getattr(getattr(cfg, "dataset", None), "num_classes", 10)) if cfg else 10
        feat_dim = getattr(getattr(cfg, "dataset", None), "feature_dim", None) if cfg else None

        if dname == "synthetic":
            base = ModelClass(num_classes=num_classes, input_dim=int(feat_dim or 60)).to(device)
        elif dname in {"shakespeare", "sent140"}:
            base = ModelClass(num_classes=num_classes, vocab_size=int(feat_dim or 10_000)).to(device)
        elif dname == "extrasensory":
            base = ModelClass(num_classes=num_classes, input_dim=int(feat_dim or 226)).to(device)
        elif dname == "har_lstm":
            base = ModelClass(num_classes=num_classes, input_size=int(feat_dim or 3)).to(device)
        elif dname in {"har", "har_mlp"}:
            base = ModelClass(num_classes=num_classes, input_dim=int(feat_dim or 47)).to(device)
        else:
            base = ModelClass(num_classes=num_classes).to(device)
        # Force CrossEntropyLoss as shared models return logits
        return (base, "Mclr_CrossEntropy")

    if model_name == "mclr":
        if dataset.lower() == "mnist":
            base = Mclr_Logistic().to(device)
        else:
            base = Mclr_Logistic(60, 10).to(device)
    elif model_name == "cnn":
        if dataset.lower() == "mnist":
            base = Net().to(device)
        elif dataset.lower() == "cifar10":
            base = CNNCifar(10).to(device)
        elif dataset.lower() == "femnist":
            base = FemnistNet().to(device)
        else:
            base = Net().to(device)
    elif model_name == "dnn":
        if dataset.lower() == "mnist":
            base = DNN().to(device)
        else:
            base = DNN(60, 20, 10).to(device)
    else:
        raise ValueError(f"Unsupported model {model_name}")
    return (base, model_name)


class _BaseWrapper:
    def __init__(self, args: Optional[Union[Dict[str, Any], SimpleNamespace]] = None, cfg: Optional[AppConfig] = None):
        self.args = SimpleNamespace(**(args or {})) if isinstance(args, dict) else (args or SimpleNamespace())
        self.cfg = cfg

    def _resolve(self, name: str, default: Any) -> Any:
        if hasattr(self.args, name):
            return getattr(self.args, name)
        if self.cfg:
            namespace = getattr(self.cfg, name.split('.')[0], None)
        return default


class PFedMeWrapper(_BaseWrapper):
    def run(self) -> None:
        cfg = self.cfg
        dataset = getattr(self.args, "dataset", cfg.dataset.name if cfg else "femnist")
        model_name = getattr(self.args, "model", "cnn")
        batch_size = getattr(self.args, "batch_size", cfg.client.batch_size if cfg else 20)
        lr = getattr(self.args, "learning_rate", cfg.server.inner_lr if cfg else 0.005)
        beta = getattr(self.args, "beta", 1.0)
        lamda = getattr(self.args, "lamda", 15)
        num_glob_iters = getattr(self.args, "num_global_iters", cfg.num_rounds if cfg else 800)
        local_epochs = getattr(self.args, "local_epochs", cfg.client.local_epochs if cfg else 20)
        optimizer = getattr(self.args, "optimizer", "SGD")
        numusers = getattr(self.args, "numusers", cfg.dataset.clients_per_round or cfg.dataset.num_clients if cfg else 20)
        K = getattr(self.args, "K", 5)
        personal_lr = getattr(self.args, "personal_learning_rate", 0.09)
        times = getattr(self.args, "times", 1)
        gpu = getattr(self.args, "gpu", 0)

        model = _build_model(dataset, model_name, cfg=cfg)
        
        # Load data via leaf_loader
        train_files = getattr(cfg.dataset, 'train_json_files', None) or getattr(cfg.dataset, 'json_files', None)
        val_files = getattr(cfg.dataset, 'val_json_files', None)
        
        train_datasets = {}
        test_datasets = {}
        
        if train_files:
            td, vd, _ = load_leaf_splits_datasets(
                root=cfg.dataset.root,
                train_files=train_files,
                val_files=val_files,
                holdout_files=None,
                limit=cfg.dataset.num_clients,
            )
            train_datasets = td
            test_datasets = vd
        else:
            splits = auto_select_leaf_splits(
                root=cfg.dataset.root,
                batch_size=batch_size,
                shuffle=True,
                num_clients=int(getattr(cfg.dataset, 'num_clients', 0) or 0),
                seed=int(getattr(cfg, 'seed', 42)),
            )
            # splits.train is dict of DataLoaders
            train_datasets = {cid: ldr.dataset for cid, ldr in splits.train.items()}
            test_datasets = {cid: ldr.dataset for cid, ldr in splits.val.items()}

        users = []
        device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
        
        # Create UserpFedMe objects
        client_ids = list(train_datasets.keys())
        for i, cid in enumerate(client_ids):
            # Extract tensors
            train_data = train_datasets[cid]
            test_data = test_datasets.get(cid)
            
            # Helper to extract x, y
            def extract_xy(ds):
                if hasattr(ds, 'tensors'):
                    return {'x': ds.tensors[0], 'y': ds.tensors[1]}
                return {'x': [], 'y': []}

            u_train = extract_xy(train_data)
            u_test = extract_xy(test_data) if test_data else {'x': [], 'y': []}
            
            user = UserpFedMe(
                device,
                i, # numeric_id
                u_train,
                u_test,
                model,
                batch_size,
                lr,
                beta,
                lamda,
                local_epochs,
                optimizer,
                K,
                personal_lr
            )
            users.append(user)

        server = PFedMeServer(
            device,
            dataset,
            "pFedMe",
            model,
            batch_size,
            lr,
            beta,
            lamda,
            num_glob_iters,
            local_epochs,
            optimizer,
            numusers,
            K,
            personal_lr,
            times,
            users=users,
        )
        server.train()
        server.test()


class PFedAvgWrapper(_BaseWrapper):
    def run(self) -> None:
        cfg = self.cfg
        dataset = getattr(self.args, "dataset", cfg.dataset.name if cfg else "femnist")
        model_name = getattr(self.args, "model", "cnn")
        batch_size = getattr(self.args, "batch_size", cfg.client.batch_size if cfg else 20)
        lr = getattr(self.args, "learning_rate", cfg.server.inner_lr if cfg else 0.005)
        beta = getattr(self.args, "beta", 1.0)
        lamda = getattr(self.args, "lamda", 15)
        num_glob_iters = getattr(self.args, "num_global_iters", cfg.num_rounds if cfg else 800)
        local_epochs = getattr(self.args, "local_epochs", cfg.client.local_epochs if cfg else 20)
        optimizer = getattr(self.args, "optimizer", "SGD")
        numusers = getattr(self.args, "numusers", cfg.dataset.clients_per_round or cfg.dataset.num_clients if cfg else 20)
        times = getattr(self.args, "times", 1)
        gpu = getattr(self.args, "gpu", 0)

        model = _build_model(dataset, model_name, cfg=cfg)
        server = PerFedAvgServer(
            torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"),
            dataset,
            "PerAvg",
            model,
            batch_size,
            lr,
            beta,
            lamda,
            num_glob_iters,
            local_epochs,
            optimizer,
            numusers,
            times,
            users=None,
        )
        server.train()
        server.test()


class PFedFedAvgWrapper(_BaseWrapper):
    def run(self) -> None:
        cfg = self.cfg
        dataset = getattr(self.args, "dataset", cfg.dataset.name if cfg else "femnist")
        model_name = getattr(self.args, "model", "cnn")
        batch_size = getattr(self.args, "batch_size", cfg.client.batch_size if cfg else 20)
        lr = getattr(self.args, "learning_rate", cfg.server.inner_lr if cfg else 0.005)
        beta = getattr(self.args, "beta", 1.0)
        lamda = getattr(self.args, "lamda", 0)
        num_glob_iters = getattr(self.args, "num_global_iters", cfg.num_rounds if cfg else 800)
        local_epochs = getattr(self.args, "local_epochs", cfg.client.local_epochs if cfg else 20)
        optimizer = getattr(self.args, "optimizer", "SGD")
        numusers = getattr(self.args, "numusers", cfg.dataset.clients_per_round or cfg.dataset.num_clients if cfg else 20)
        times = getattr(self.args, "times", 1)
        gpu = getattr(self.args, "gpu", 0)

        model = _build_model(dataset, model_name, cfg=cfg)
        server = FedAvgServer(
            torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"),
            dataset,
            "FedAvg",
            model,
            batch_size,
            lr,
            beta,
            lamda,
            num_glob_iters,
            local_epochs,
            optimizer,
            numusers,
            times,
            users=None,
        )
        server.train()
        server.test()