#!/usr/bin/env python
import h5py
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
import argparse
import importlib
import random
import os
import wandb
from FLAlgorithms.servers.serveravg import FedAvg as ServerFedAvg
from FLAlgorithms.servers.serverpFedMe import pFedMe as ServerPFedMe
from FLAlgorithms.servers.serverperavg import PerAvg as ServerPerAvg
from FLAlgorithms.trainmodel.models import *
from utils.plot_utils import *
import torch
from src.core import AppConfig
from src.utils.seed import set_global_seed
from src.data.leaf_loader import load_leaf_datasets, load_leaf_splits_datasets, auto_select_leaf_splits

def _init_seeds(cfg: AppConfig | None) -> int:
    seed = 42
    try:
        if cfg is not None and hasattr(cfg, "seed"):
            seed = int(getattr(cfg, "seed", 42))
    except Exception:
        seed = 42
    set_global_seed(seed, deterministic=False)
    return seed

def main(dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
         local_epochs, optimizer, numusers, K, personal_learning_rate, times, gpu, cfg: AppConfig = None):

    # Initialize seeds early for reproducibility
    run_seed = _init_seeds(cfg)

    # Get device status: Check GPU or CPU
    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")

    # Preserve model name separately to avoid shadowing when wrapping as tuple later
    model_name = model

    # Load datasets if cfg is provided
    users = []
    if cfg:
        print(f"Loading data from {cfg.dataset.root}...")
        # Unified train/val/holdout routing via leaf_loader
        train_files = getattr(cfg.dataset, "train_json_files", None) or getattr(cfg.dataset, "json_files", None)
        val_files = getattr(cfg.dataset, "val_json_files", None)
        holdout_files = getattr(cfg.dataset, "holdout_json_files", None)
        if train_files:
            train_datasets, test_datasets, holdout_datasets = load_leaf_splits_datasets(
                root=cfg.dataset.root,
                train_files=train_files,
                val_files=val_files,
                holdout_files=holdout_files,
                limit=cfg.dataset.num_clients,
            )
        else:
            # Auto-select clients across all files with deterministic sampling
            splits = auto_select_leaf_splits(
                root=cfg.dataset.root,
                batch_size=cfg.client.batch_size,
                shuffle=True,
                num_clients=int(getattr(cfg.dataset, 'num_clients', 0) or 0),
                seed=int(getattr(cfg, 'seed', 42)),
                holdout_limit=int(getattr(cfg.dataset, 'holdout_client_limit', 0) or 0) or None,
            )
            train_datasets = {cid: ldr.dataset for cid, ldr in splits.train.items()}
            test_datasets = {cid: ldr.dataset for cid, ldr in splits.val.items()}
            holdout_datasets = splits.holdout
        if holdout_datasets:
            print(f"Loaded {len(holdout_datasets)} holdout clients")

        # Create User objects
        # We need to import UserpFedMe here or pass it
        from FLAlgorithms.users.userpFedMe import UserpFedMe
        from FLAlgorithms.users.useravg import UserAVG
        from FLAlgorithms.users.userperavg import UserPerAvg
        from src.client.models import get_model_class
        
        client_ids = list(train_datasets.keys())
        total_users_count = len(client_ids)

        # Resolve full-participation semantics for pFedMe-family baselines.
        # If numusers is unspecified/0/negative or larger than available clients, use all clients.
        try:
            raw_numusers = int(numusers) if numusers is not None else 0
        except Exception:
            raw_numusers = 0
        if raw_numusers <= 0 or raw_numusers >= total_users_count:
            numusers = total_users_count

        for i, cid in enumerate(client_ids):
            train_data = train_datasets[cid]
            test_data = test_datasets.get(cid, [])
            
            # Create a dummy model for initialization (will be overwritten by server)
            # We need to know the model class.
            # For now, we rely on the 'model' string argument passed to main
            # But we need to instantiate it to pass to User.
            
            # ... (Model instantiation logic duplicated from below) ...
            if(model_name == "mclr"):
                if(dataset == "Mnist"):
                    model_obj = Mclr_Logistic().to(device)
                else:
                    model_obj = Mclr_Logistic(60,50).to(device)
            elif(model_name == "cnn"):
                # Try to load from src.client.models first
                ModelClass = get_model_class(dataset)
                if ModelClass:
                    print(f"[pFedMe/Per-FedAvg] Initializing model: {ModelClass.__name__} for dataset: {dataset}")
                    if dataset == "sent140":
                        # Sent140 needs vocab_size
                        vocab_size = getattr(cfg.dataset, "feature_dim", 10000)
                        model_obj = ModelClass(num_classes=cfg.dataset.num_classes, vocab_size=vocab_size).to(device)
                    elif dataset == "synthetic":
                        # Synthetic needs input_dim
                        input_dim = getattr(cfg.dataset, "feature_dim", 60)
                        model_obj = ModelClass(num_classes=cfg.dataset.num_classes, input_dim=input_dim).to(device)
                    elif dataset == "extrasensory":
                        input_dim = getattr(cfg.dataset, "feature_dim", 226)
                        model_obj = ModelClass(num_classes=cfg.dataset.num_classes, input_dim=input_dim).to(device)
                    elif dataset == "har_lstm":
                        input_size = getattr(cfg.dataset, "feature_dim", 3)
                        model_obj = ModelClass(num_classes=cfg.dataset.num_classes, input_size=input_size).to(device)
                    elif dataset in {"har", "har_mlp"}:
                        input_dim = getattr(cfg.dataset, "feature_dim", 47)
                        model_obj = ModelClass(num_classes=cfg.dataset.num_classes, input_dim=input_dim).to(device)
                    else:
                        model_obj = ModelClass(num_classes=cfg.dataset.num_classes).to(device)
                elif(dataset == "Mnist"):
                    model_obj = Net().to(device)
                elif(dataset == "Cifar10"):
                    model_obj = CNNCifar(10).to(device)
                elif(dataset == "femnist"):
                    model_obj = FemnistNet().to(device)
            elif(model_name == "dnn"):
                if(dataset == "Mnist"):
                    model_obj = DNN().to(device)
                else: 
                    model_obj = DNN(60,256,50).to(device)
            
            # Wrap model in tuple as expected by User classes in pFedMe
            # Force "Mclr_CrossEntropy" if using src.client.models to ensure CrossEntropyLoss
            if ModelClass:
                model_tuple = (model_obj, "Mclr_CrossEntropy")
            else:
                model_tuple = (model_obj, model_name)

            if algorithm == "pFedMe":
                user = UserpFedMe(
                    device, cid, train_data, test_data, model_tuple,
                    batch_size, learning_rate, beta, lamda, local_epochs,
                    optimizer, K, personal_learning_rate
                )
            elif algorithm == "FedAvg":
                user = UserAVG(
                    device, cid, train_data, test_data, model_tuple,
                    batch_size, learning_rate, beta, lamda, local_epochs, optimizer
                )
            elif algorithm == "PerAvg":
                user = UserPerAvg(
                    device, cid, train_data, test_data, model_tuple,
                    batch_size, learning_rate, beta, lamda, local_epochs, optimizer,
                    total_users_count, numusers
                )
            
            users.append(user)

    for i in range(times):
        print("---------------Running time:------------",i)
        
        # Initialize WandB: prefer explicit env, else if Per-FedAvg use canonical name
        display_algo_name = "PerFedAvg" if algorithm == "PerAvg" else algorithm
        # If user set WANDB_NAME keep it; else if experiment_name provided AND not PerAvg override request, use it
        env_run_name = os.environ.get("WANDB_NAME")
        cfg_run_name = None
        if cfg is not None:
            cfg_run_name = getattr(cfg, "experiment_name", None)
        # For Per-FedAvg we force algorithm-only name unless env overrides
        if env_run_name:
            run_name = env_run_name
        elif algorithm == "PerAvg":
            run_name = display_algo_name
        elif cfg_run_name:
            run_name = cfg_run_name
        else:
            run_name = f"{display_algo_name}_{dataset}_{num_glob_iters}r_{numusers}c"
        run_id = os.environ.get("WANDB_RUN_ID")
        wandb.init(
            project=cfg.wandb_project if cfg and hasattr(cfg, 'wandb_project') else "HyperQLoRA-HFL",
            name=run_name,
            id=run_id,
            resume="allow" if run_id else None,
            config={
                "dataset": {
                    "name": dataset,
                },
                "client": {
                    "batch_size": batch_size,
                    "local_epochs": local_epochs,
                    "learning_rate": learning_rate,
                    "personal_learning_rate": personal_learning_rate,
                },
                "server": {
                    "num_global_iters": num_glob_iters,
                    "num_clients_per_round": numusers,
                    "algorithm": algorithm,
                    "beta": beta,
                    "lamda": lamda,
                    "K": K,
                },
                "experiment_name": run_name,
                "seed": run_seed,
                "model": model,
                "optimizer": optimizer,
                "times": times,
                "gpu": gpu
            },
            reinit=True
        )

        # Generate model
        # Use the same logic as above to ensure consistency
        from src.client.models import get_model_class
        ModelClass = get_model_class(dataset)
        
        if ModelClass:
            if dataset == "sent140":
                vocab_size = getattr(cfg.dataset, "feature_dim", 10000)
                model_obj = ModelClass(num_classes=cfg.dataset.num_classes, vocab_size=vocab_size).to(device)
            elif dataset == "synthetic":
                input_dim = getattr(cfg.dataset, "feature_dim", 60)
                model_obj = ModelClass(num_classes=cfg.dataset.num_classes, input_dim=input_dim).to(device)
            elif dataset == "extrasensory":
                input_dim = getattr(cfg.dataset, "feature_dim", 226)
                model_obj = ModelClass(num_classes=cfg.dataset.num_classes, input_dim=input_dim).to(device)
            elif dataset == "har_lstm":
                input_size = getattr(cfg.dataset, "feature_dim", 3)
                model_obj = ModelClass(num_classes=cfg.dataset.num_classes, input_size=input_size).to(device)
            elif dataset in {"har", "har_mlp"}:
                input_dim = getattr(cfg.dataset, "feature_dim", 47)
                model_obj = ModelClass(num_classes=cfg.dataset.num_classes, input_dim=input_dim).to(device)
            else:
                model_obj = ModelClass(num_classes=cfg.dataset.num_classes).to(device)
            # Force "Mclr_CrossEntropy" to ensure CrossEntropyLoss is used for logits
            model = (model_obj, "Mclr_CrossEntropy")
        elif(model_name == "mclr"):
            if(dataset == "Mnist"):
                model = (Mclr_Logistic().to(device), model_name)
            else:
                model = (Mclr_Logistic(60,10).to(device), model_name)
                
        elif(model_name == "cnn"):
            if(dataset == "Mnist"):
                model = (Net().to(device), model_name)
            elif(dataset == "Cifar10"):
                model = (CNNCifar(10).to(device), model_name)
            elif(dataset == "femnist"):
                model = (FemnistNet().to(device), model_name)
            
        elif(model_name == "dnn"):
            if(dataset == "Mnist"):
                model = (DNN().to(device), model_name)
            else: 
                model = (DNN(60,20,10).to(device), model_name)

        # select algorithm
        if(algorithm == "FedAvg"):
            server = ServerFedAvg(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_epochs, optimizer, numusers, i, users=users)
        
        if(algorithm == "pFedMe"):
            server = ServerPFedMe(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_epochs, optimizer, numusers, K, personal_learning_rate, i, users=users)

        if(algorithm == "PerAvg"):
            server = ServerPerAvg(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_epochs, optimizer, numusers, i, users=users)

        # Inject save_dir from cfg if available
        server.save_dir = cfg.save_dir if cfg and hasattr(cfg, 'save_dir') else "save"

        server.train()
        server.test()
        
        if cfg and holdout_datasets:
            print("Evaluating on holdout data...")
            from FLAlgorithms.users.userpFedMe import UserpFedMe
            from FLAlgorithms.users.useravg import UserAVG
            from FLAlgorithms.users.userperavg import UserPerAvg
            
            holdout_users = []
            for cid, data in holdout_datasets.items():
                # Create model instance for user
                if(model_name == "mclr"):
                    if(dataset == "Mnist"):
                        model_obj = Mclr_Logistic().to(device)
                    else:
                        model_obj = Mclr_Logistic(60,10).to(device)
                elif(model_name == "cnn"):
                    if(dataset == "Mnist"):
                        model_obj = Net().to(device)
                    elif(dataset == "Cifar10"):
                        model_obj = CNNCifar(10).to(device)
                    elif(dataset == "femnist"):
                        model_obj = FemnistNet().to(device)
                elif(model_name == "dnn"):
                    if(dataset == "Mnist"):
                        model_obj = DNN().to(device)
                    else: 
                        model_obj = DNN(60,20,10).to(device)
                
                model_tuple = (model_obj, model_name)
                
                # Create user
                # Note: We use server.model (global model) for evaluation
                # But User init takes model_tuple.
                # We will set parameters later.
                
                if algorithm == "pFedMe":
                    user = UserpFedMe(
                        device, cid, data, [], model_tuple,
                        batch_size, learning_rate, beta, lamda, local_epochs,
                        optimizer, K, personal_learning_rate
                    )
                elif algorithm == "FedAvg":
                    user = UserAVG(
                        device, cid, data, [], model_tuple,
                        batch_size, learning_rate, beta, lamda, local_epochs, optimizer
                    )
                elif algorithm == "PerAvg":
                    # For holdout, approximate totals with number of holdout clients
                    holdout_total = len(holdout_datasets)
                    user = UserPerAvg(
                        device, cid, data, [], model_tuple,
                        batch_size, learning_rate, beta, lamda, local_epochs, optimizer,
                        holdout_total, min(numusers, holdout_total)
                    )
                
                # Set user model to server's global model
                user.set_parameters(server.model)
                holdout_users.append(user)

            total_acc = 0
            total_loss = 0
            total_samples = 0
            
            rows = []
            losses = []
            accs = []

            for user in holdout_users:
                # User.test() returns (correct_count, loss_sum, num_samples) based on serverbase.py usage
                # But let's verify if User.test() returns that.
                # In serverbase.py: ct, cl, ns = c.test()
                # So yes.
                ct, cl, ns = user.test()
                
                acc = ct / ns if ns > 0 else 0
                loss = cl / ns if ns > 0 else 0
                
                rows.append((user.id, ns, loss, acc))
                losses.append(loss)
                accs.append(acc)
                
                total_acc += ct
                total_loss += cl
                total_samples += ns
            
            mean_loss = total_loss / max(total_samples, 1)
            mean_acc = total_acc / max(total_samples, 1)
            
            log_data = {
                "holdout/loss_mean": mean_loss,
                "holdout/loss_std": float(np.std(losses)) if len(losses) > 1 else 0.0,
                "holdout/loss_min": min(losses) if losses else 0,
                "holdout/loss_max": max(losses) if losses else 0,
                "holdout/acc_mean": mean_acc,
                "holdout/acc_std": float(np.std(accs)) if len(accs) > 1 else 0.0,
                "holdout/acc_min": min(accs) if accs else 0,
                "holdout/acc_max": max(accs) if accs else 0,
                "holdout/num_clients": len(rows),
                "holdout/num_samples": total_samples,
            }
            
            table = wandb.Table(columns=["client_id", "samples", "loss", "accuracy"])
            for row in rows:
                table.add_data(*row)
            log_data["holdout/client_metrics"] = table
            wandb.log(log_data)
            # Persist standardized per-client holdout CSV under algorithm-specific directory
            try:
                algo_dir = "pfedme" if algorithm == "pFedMe" else ("perfedavg" if algorithm == "PerAvg" else algorithm.lower())
                base_dir = cfg.save_dir if cfg and hasattr(cfg, 'save_dir') else "save"
                out_dir = Path(base_dir) / algo_dir
                out_dir.mkdir(parents=True, exist_ok=True)
                std_rows = []
                for (cid, ns, loss, acc) in rows:
                    std_rows.append({
                        'client_id': cid,
                        'num_samples': int(ns) if ns is not None else None,
                        'holdout_loss': float(loss),
                        'holdout_acc': float(acc),
                    })
                if std_rows:
                    std_df = pd.DataFrame(std_rows)
                    csv_path = out_dir / "holdout.csv"
                    std_df.to_csv(csv_path, index=False)
                    print(f"Saved standardized holdout per-client metrics to {csv_path}")
            except Exception as e:
                print(f"[pFedMe] Failed to save standardized holdout per-client metrics: {e}")
            print(f"Holdout Evaluation: Loss={mean_loss:.4f}, Acc={mean_acc:.4f} (Clients={len(rows)})")

        wandb.finish()


    # Average data (legacy .h5-based). Skip when disabled for integrated runs.
    if os.getenv("PFEDME_DISABLE_H5", "1") == "1":
        return
    # Average data 
    if(algorithm == "PerAvg"):
        algorithm == "PerAvg_p"
    if(algorithm == "pFedMe"):
        average_data(num_users=numusers, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=lamda,learning_rate=learning_rate, beta = beta, algorithms="pFedMe_p", batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate,times = times)
    average_data(num_users=numusers, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=lamda,learning_rate=learning_rate, beta = beta, algorithms=algorithm, batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate,times = times)

class pFedMe:
    """
    Wrapper for pFedMe to be used with ExtAdapter.
    """
    def __init__(self, args, cfg=None):
        self.args = args
        self.cfg = cfg

    def run(self):
        # Map args/cfg to pFedMe main parameters
        # Priority: self.args (specific) > self.cfg (general) > defaults
        dataset = getattr(self.args, 'dataset', self.cfg.dataset.name if self.cfg else 'femnist')
        algorithm = getattr(self.args, 'algorithm', 'pFedMe')
        model = getattr(self.args, 'model', 'cnn')
        batch_size = getattr(self.args, 'batch_size', self.cfg.client.batch_size if self.cfg else 32)
        learning_rate = getattr(self.args, 'learning_rate', self.cfg.server.inner_lr if self.cfg else 0.005)
        beta = getattr(self.args, 'beta', 1.0)
        lamda = getattr(self.args, 'lamda', 15)
        num_glob_iters = getattr(self.args, 'num_global_iters', self.cfg.num_rounds if self.cfg else 800)
        local_epochs = getattr(self.args, 'local_epochs', self.cfg.client.local_epochs if self.cfg else 20)
        optimizer = getattr(self.args, 'optimizer', 'SGD')
        numusers = getattr(self.args, 'numusers', self.cfg.dataset.clients_per_round if self.cfg and self.cfg.dataset.clients_per_round else 20)
        K = getattr(self.args, 'K', 5)  # Personalization steps
        personal_learning_rate = getattr(self.args, 'personal_learning_rate', 0.09)
        times = getattr(self.args, 'times', 1)
        gpu = getattr(self.args, 'gpu', 0)

        # Call main once with resolved parameters
        main(
            dataset,
            algorithm,
            model,
            batch_size,
            learning_rate,
            beta,
            lamda,
            num_glob_iters,
            local_epochs,
            optimizer,
            numusers,
            K,
            personal_learning_rate,
            times,
            gpu,
            cfg=self.cfg,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Cifar10", choices=["Mnist", "Synthetic", "Cifar10", "femnist"])
    parser.add_argument("--model", type=str, default="cnn", choices=["dnn", "mclr", "cnn"])
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=0.005, help="Local learning rate")
    parser.add_argument("--beta", type=float, default=1.0, help="Average moving parameter for pFedMe, or Second learning rate of Per-FedAvg")
    parser.add_argument("--lamda", type=int, default=15, help="Regularization term")
    parser.add_argument("--num_global_iters", type=int, default=800)
    parser.add_argument("--local_epochs", type=int, default=20)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--algorithm", type=str, default="pFedMe",choices=["pFedMe", "PerAvg", "FedAvg"]) 
    parser.add_argument("--numusers", type=int, default=20, help="Number of Users per round")
    parser.add_argument("--K", type=int, default=5, help="Computation steps")
    parser.add_argument("--personal_learning_rate", type=float, default=0.09, help="Persionalized learning rate to caculate theta aproximately using K steps")
    parser.add_argument("--times", type=int, default=5, help="running time")
    parser.add_argument("--gpu", type=int, default=0, help="Which GPU to run the experiments, -1 mean CPU, 0,1,2 for GPU")
    args = parser.parse_args()

    print("=" * 80)
    print("Summary of training process:")
    print("Algorithm: {}".format(args.algorithm))
    print("Batch size: {}".format(args.batch_size))
    print("Learing rate       : {}".format(args.learning_rate))
    print("Average Moving       : {}".format(args.beta))
    print("Subset of users      : {}".format(args.numusers))
    print("Number of global rounds       : {}".format(args.num_global_iters))
    print("Number of local rounds       : {}".format(args.local_epochs))
    print("Dataset       : {}".format(args.dataset))
    print("Local Model       : {}".format(args.model))
    print("=" * 80)

    main(
        dataset=args.dataset,
        algorithm = args.algorithm,
        model=args.model,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        beta = args.beta, 
        lamda = args.lamda,
        num_glob_iters=args.num_global_iters,
        local_epochs=args.local_epochs,
        optimizer= args.optimizer,
        numusers = args.numusers,
        K=args.K,
        personal_learning_rate=args.personal_learning_rate,
        times = args.times,
        gpu=args.gpu
        )
