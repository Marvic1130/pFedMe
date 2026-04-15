import torch
import os
import time
import wandb
from pathlib import Path
import pandas as pd

from FLAlgorithms.users.userpFedMe import UserpFedMe
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data
import numpy as np
 
# Implementation for pFedMe Server

class pFedMe(Server):
    def __init__(self, device,  dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                 local_epochs, optimizer, num_users, K, personal_learning_rate, times, users=None):
        super().__init__(device, dataset,algorithm, model[0], batch_size, learning_rate, beta, lamda, num_glob_iters,
                         local_epochs, optimizer, num_users, times)

        self.K = K
        self.personal_learning_rate = personal_learning_rate

        if users is not None:
            self.users = users
            self.total_train_samples = sum([u.train_samples for u in self.users])
        else:
            # Initialize data for all  users
            data = read_data(dataset)
            total_users = len(data[0])
            for i in range(total_users):
                id, train , test = read_user_data(i, data, dataset)
                user = UserpFedMe(device, id, train, test, model, batch_size, learning_rate, beta, lamda, local_epochs, optimizer, K, personal_learning_rate)
                self.users.append(user)
                self.total_train_samples += user.train_samples
            print("Number of users / total users:",num_users, " / " ,total_users)
            
        print("Finished creating pFedMe server.")

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
        self.send_parameters()
        for glob_iter in range(self.num_glob_iters):
            round_start_time = time.time()
            print("-------------Round number: ",glob_iter, " -------------")
            # send all parameter for users 
            # self.send_parameters() # Moved to end

            # Evaluate gloal model on user for each interation
            # print("Evaluate global model")
            # print("")
            # self.evaluate(glob_iter) # Moved to end

            client_durations = []
            # do update for all users not only selected users
            for user in self.users:
                client_start_time = time.time()
                user.train(self.local_epochs) #* user.train_samples
                client_durations.append(time.time() - client_start_time)
            
            # choose several users to send back upated model to server
            # self.personalized_evaluate()
            self.selected_users = self.select_users(glob_iter,self.num_users)

            # Evaluate gloal model on user for each interation
            #print("Evaluate persionalized model")
            #print("")
            # Log personalized model metrics (no commit to batch with global metrics)
            self.evaluate_personalized_model(glob_iter, commit=False)
            #self.aggregate_parameters()
            
            server_start_time = time.time()
            self.persionalized_aggregate_parameters()
            server_duration = time.time() - server_start_time
            
            self.send_parameters()
            
            wall_clock_duration = time.time() - round_start_time

            max_client_duration = max(client_durations) if client_durations else 0.0
            mean_client_duration = np.mean(client_durations) if client_durations else 0.0
            simulated_duration = max_client_duration + server_duration

            # Log global model metrics with unified duration fields
            self.evaluate(
                glob_iter,
                commit=True,
                extra_metrics={
                    "client_duration": mean_client_duration,
                    "max_client_duration": max_client_duration,
                    "server_duration": server_duration,
                    "simulated_duration": simulated_duration,
                    "wall_clock_duration": wall_clock_duration,
                },
                duration=simulated_duration,
            )


        #print(loss)
        self.save_results()
        self.save_model()
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
            df = pd.DataFrame(rows)
            base_dir = getattr(self, 'save_dir', 'save')
            out_dir = Path(base_dir) / 'pfedme'
            out_dir.mkdir(parents=True, exist_ok=True)
            csv_path = out_dir / 'final.csv'
            df.to_csv(csv_path, index=False)
            print(f"Saved pFedMe final per-client metrics to {csv_path}")
        except Exception as e:
            print(f"[pFedMe] Failed to save final per-client metrics: {e}")


