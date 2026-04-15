import torch
import os
import time
import wandb
import numpy as np
from pathlib import Path
import pandas as pd

from FLAlgorithms.users.userperavg import UserPerAvg
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data

# Implementation for per-FedAvg Server

class PerAvg(Server):
    def __init__(self,device, dataset,algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                 local_epochs, optimizer, num_users,times, users=None):
        super().__init__(device, dataset,algorithm, model[0], batch_size, learning_rate, beta, lamda, num_glob_iters,
                         local_epochs, optimizer, num_users, times)

        if users is not None:
            # Use provided users (pre-built with correct total_users/num_users)
            self.users = users
            self.total_train_samples = sum([u.train_samples for u in self.users])
            total_users = len(self.users)
            print("Number of users / total users:", num_users, " / ", total_users)
            print("Finished creating Local Per-Avg (from provided users).")
        else:
            # Initialize data for all users from disk
            data = read_data(dataset)
            total_users = len(data[0])
            for i in range(total_users):
                id, train , test = read_user_data(i, data, dataset)
                user = UserPerAvg(device, id, train, test, model, batch_size, learning_rate, beta, lamda, local_epochs, optimizer ,total_users , num_users)
                self.users.append(user)
                self.total_train_samples += user.train_samples
            print("Number of users / total users:",num_users, " / ",total_users)
            print("Finished creating Local Per-Avg.")

    def send_grads(self):
        assert (self.users is not None and len(self.users) > 0)
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad)
        for user in self.users:
            user.set_grads(grads)

    def train(self):
        loss = []
        for glob_iter in range(self.num_glob_iters):
            start_time = time.time()
            print("-------------Round number: ",glob_iter, " -------------")
            # send all parameter for users 
            self.send_parameters()

            # Evaluate gloal model on user for each interation
            print("Evaluate global model with one step update")
            print("")
            self.evaluate_one_step()

            client_durations = []
            # choose several users to send back upated model to server
            self.selected_users = self.select_users(glob_iter,self.num_users)
            for user in self.selected_users:
                client_start_time = time.time()
                user.train(self.local_epochs) #* user.train_samples
                client_durations.append(time.time() - client_start_time)
                
            server_start_time = time.time()
            self.aggregate_parameters()
            server_duration = time.time() - server_start_time
            
            wall_clock_duration = time.time() - start_time
            
            max_client_duration = max(client_durations) if client_durations else 0.0
            mean_client_duration = np.mean(client_durations) if client_durations else 0.0
            simulated_duration = max_client_duration + server_duration

            # Use standardized logging via evaluate() for parity with other baselines
            self.evaluate(
                glob_iter,
                commit=True,
                extra_metrics={
                    "num_selected_users": int(len(self.selected_users)) if self.selected_users is not None else 0,
                    "num_total_users": int(len(self.users)) if self.users is not None else 0,
                    "client_duration": float(mean_client_duration),
                    "max_client_duration": float(max_client_duration),
                    "server_duration": float(server_duration),
                    "simulated_duration": float(simulated_duration),
                    "wall_clock_duration": float(wall_clock_duration),
                },
                duration=float(simulated_duration),
            )

        self.save_results()
        self.save_model()
        # Persist final per-client test metrics for fair comparison
        try:
            ids, num_samples, tot_correct, losses = self.test()
            rows = []
            for i, cid in enumerate(ids):
                ns = num_samples[i]
                correct = tot_correct[i]
                total_loss = losses[i]
                avg_loss = total_loss / ns if ns > 0 else 0.0
                acc = correct / ns if ns > 0 else 0.0
                rows.append({
                    'client_id': cid,
                    'num_samples': ns,
                    'test_loss': avg_loss,
                    'test_acc': acc,
                })
            if rows:
                df = pd.DataFrame(rows)
                base_dir = getattr(self, 'save_dir', 'save')
                out_dir = Path(base_dir) / 'perfedavg'
                out_dir.mkdir(parents=True, exist_ok=True)
                csv_path = out_dir / 'final.csv'
                df.to_csv(csv_path, index=False)
                print(f"Saved Per-FedAvg final per-client metrics to {csv_path}")
        except Exception as e:
            print(f"[PerFedAvg] Failed to save final per-client metrics: {e}")
