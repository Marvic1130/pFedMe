import torch
import os
import time

from FLAlgorithms.users.useravg import UserAVG
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data
import numpy as np
import wandb

# Implementation for FedAvg Server

class FedAvg(Server):
    def __init__(self, device, dataset,algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                 local_epochs, optimizer, num_users, times, users=None):
        super().__init__(device, dataset,algorithm, model[0], batch_size, learning_rate, beta, lamda, num_glob_iters,
                         local_epochs, optimizer, num_users, times)

        if users is not None:
            # Use provided users
            self.users = users
            self.total_train_samples = sum([u.train_samples for u in self.users])
        else:
            # Initialize data for all  users
            data = read_data(dataset)
            total_users = len(data[0])
            for i in range(total_users):
                id, train , test = read_user_data(i, data, dataset)
                user = UserAVG(device, id, train, test, model, batch_size, learning_rate,beta,lamda, local_epochs, optimizer)
                self.users.append(user)
                self.total_train_samples += user.train_samples
            
            print("Number of users / total users:",num_users, " / " ,total_users)
            
        print("Finished creating FedAvg server.")

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
            start_time = time.time()
            print("-------------Round number: ",glob_iter, " -------------")
            #loss_ = 0
            # self.send_parameters() # Moved to end of loop

            # Evaluate model each interation
            # self.evaluate(glob_iter) # Moved to end of loop

            client_durations = []
            self.selected_users = self.select_users(glob_iter,self.num_users)
            for user in self.selected_users:
                client_start_time = time.time()
                user.train(self.local_epochs) #* user.train_samples
                client_durations.append(time.time() - client_start_time)
            
            server_start_time = time.time()
            self.aggregate_parameters()
            server_duration = time.time() - server_start_time
            
            self.send_parameters() # Update users with new global model for evaluation
            
            duration = time.time() - start_time
            
            max_client_duration = max(client_durations) if client_durations else 0.0
            mean_client_duration = np.mean(client_durations) if client_durations else 0.0
            simulated_duration = max_client_duration + server_duration
            
            wandb.log({
                "duration": duration,
                "client_duration": mean_client_duration,
                "max_client_duration": max_client_duration,
                "server_duration": server_duration,
                "simulated_duration": simulated_duration
            }, step=glob_iter, commit=False)
            
            self.evaluate(glob_iter)
            
            #loss_ /= self.total_train_samples
            #loss.append(loss_)
            #print(loss_)
        #print(loss)
        self.save_results()
        self.save_model()