import torch
import os
import numpy as np
import h5py
from pathlib import Path
import pandas as pd
from utils.model_utils import Metrics
import copy
import wandb

from src.utils.log_helpers import build_round_log

class Server:
    def __init__(self, device, dataset,algorithm, model, batch_size, learning_rate ,beta, lamda,
                 num_glob_iters, local_epochs, optimizer,num_users, times):

        # Set up the main attributes
        self.device = device
        self.dataset = dataset
        self.num_glob_iters = num_glob_iters
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.total_train_samples = 0
        self.model = copy.deepcopy(model)
        self.users = []
        self.selected_users = []
        self.num_users = num_users
        self.beta = beta
        self.lamda = lamda
        self.algorithm = algorithm
        self.rs_train_acc, self.rs_train_loss, self.rs_glob_acc,self.rs_train_acc_per, self.rs_train_loss_per, self.rs_glob_acc_per = [], [], [], [], [], []
        self.times = times
        # Initialize the server's grads to zeros
        #for param in self.model.parameters():
        #    param.data = torch.zeros_like(param.data)
        #    param.grad = torch.zeros_like(param.data)
        #self.send_parameters()
        
    def aggregate_grads(self):
        assert (self.users is not None and len(self.users) > 0)
        for param in self.model.parameters():
            param.grad = torch.zeros_like(param.data)
        for user in self.users:
            self.add_grad(user, user.train_samples / self.total_train_samples)

    def add_grad(self, user, ratio):
        user_grad = user.get_grads()
        for idx, param in enumerate(self.model.parameters()):
            param.grad = param.grad + user_grad[idx].clone() * ratio

    def send_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for user in self.users:
            user.set_parameters(self.model)

    def add_parameters(self, user, ratio):
        model = self.model.parameters()
        for server_param, user_param in zip(self.model.parameters(), user.get_parameters()):
            server_param.data = server_param.data + user_param.data.clone() * ratio

    def aggregate_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        total_train = 0
        #if(self.num_users = self.to)
        for user in self.selected_users:
            total_train += user.train_samples
        for user in self.selected_users:
            self.add_parameters(user, user.train_samples / total_train)

    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "server" + ".pt"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset, "server" + ".pt")
        assert (os.path.exists(model_path))
        self.model = torch.load(model_path)

    def model_exists(self):
        return os.path.exists(os.path.join("models", self.dataset, "server" + ".pt"))
    
    def select_users(self, round, num_users):
        '''selects num_clients clients weighted by number of samples from possible_clients
        Args:
            num_clients: number of clients to select; default 20
                note that within function, num_clients is set to
                min(num_clients, len(possible_clients))
        
        Return:
            list of selected clients objects
        '''
        if(num_users == len(self.users)):
            print("All users are selected")
            return self.users

        num_users = min(num_users, len(self.users))
        #np.random.seed(round)
        return np.random.choice(self.users, num_users, replace=False) #, p=pk)

    # define function for persionalized agegatation.
    def persionalized_update_parameters(self,user, ratio):
        # only argegate the local_weight_update
        for server_param, user_param in zip(self.model.parameters(), user.local_weight_updated):
            server_param.data = server_param.data + user_param.data.clone() * ratio


    def persionalized_aggregate_parameters(self):
        assert (self.users is not None and len(self.users) > 0)

        # store previous parameters
        previous_param = copy.deepcopy(list(self.model.parameters()))
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        total_train = 0
        #if(self.num_users = self.to)
        for user in self.selected_users:
            total_train += user.train_samples

        for user in self.selected_users:
            self.add_parameters(user, user.train_samples / total_train)
            #self.add_parameters(user, 1 / len(self.selected_users))

        # aaggregate avergage model with previous model using parameter beta 
        for pre_param, param in zip(previous_param, self.model.parameters()):
            param.data = (1 - self.beta)*pre_param.data + self.beta*param.data
            
    # Save loss, accurancy to h5 fiel
    def save_results(self):
        # Optional: disable legacy .h5 result files (default: disabled in this repo)
        if os.getenv("PFEDME_DISABLE_H5", "1") == "1":
            # Convergence/final CSVs are already handled elsewhere under save/<algo>/
            return
        # Ensure results directory exists to avoid FileNotFoundError
        os.makedirs("./results", exist_ok=True)
        alg = self.dataset + "_" + self.algorithm
        alg = alg + "_" + str(self.learning_rate) + "_" + str(self.beta) + "_" + str(self.lamda) + "_" + str(self.num_users) + "u" + "_" + str(self.batch_size) + "b" + "_" + str(self.local_epochs)
        if(self.algorithm == "pFedMe" or self.algorithm == "pFedMe_p"):
            alg = alg + "_" + str(self.K) + "_" + str(self.personal_learning_rate)
        alg = alg + "_" + str(self.times)
        if (len(self.rs_glob_acc) != 0 &  len(self.rs_train_acc) & len(self.rs_train_loss)) :
            with h5py.File("./results/"+'{}.h5'.format(alg, self.local_epochs), 'w') as hf:
                hf.create_dataset('rs_glob_acc', data=self.rs_glob_acc)
                hf.create_dataset('rs_train_acc', data=self.rs_train_acc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
                hf.close()
        
        # store persionalized value
        alg = self.dataset + "_" + self.algorithm + "_p"
        alg = alg  + "_" + str(self.learning_rate) + "_" + str(self.beta) + "_" + str(self.lamda) + "_" + str(self.num_users) + "u" + "_" + str(self.batch_size) + "b"+ "_" + str(self.local_epochs)
        if(self.algorithm == "pFedMe" or self.algorithm == "pFedMe_p"):
            alg = alg + "_" + str(self.K) + "_" + str(self.personal_learning_rate)
        alg = alg + "_" + str(self.times)
        if (len(self.rs_glob_acc_per) != 0 &  len(self.rs_train_acc_per) & len(self.rs_train_loss_per)) :
            with h5py.File("./results/"+'{}.h5'.format(alg, self.local_epochs), 'w') as hf:
                hf.create_dataset('rs_glob_acc', data=self.rs_glob_acc_per)
                hf.create_dataset('rs_train_acc', data=self.rs_train_acc_per)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss_per)
                hf.close()

    def test(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct, cl, ns = c.test()
            tot_correct.append(ct*1.0)
            losses.append(cl*1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct, losses

    def train_error_and_loss(self):
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct, cl, ns = c.train_error_and_loss() 
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(cl*1.0)
        
        ids = [c.id for c in self.users]
        #groups = [c.group for c in self.clients]

        return ids, num_samples, tot_correct, losses

    def test_persionalized_model(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct, cl, ns = c.test_persionalized_model()
            tot_correct.append(ct*1.0)
            losses.append(cl*1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct, losses

    def train_error_and_loss_persionalized_model(self):
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct, cl, ns = c.train_error_and_loss_persionalized_model() 
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(cl*1.0)
        
        ids = [c.id for c in self.users]
        #groups = [c.group for c in self.clients]

        return ids, num_samples, tot_correct, losses

    def get_model_size(self):
        return sum(p.numel() for p in self.model.parameters()) * 4

    def calculate_stats(self, values):
        if not values:
            return {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
        return {
            "mean": np.mean(values),
            "min": np.min(values),
            "max": np.max(values),
            "std": np.std(values),
        }

    def evaluate(self, round_idx=None, commit=True, *, extra_metrics=None, duration=None):
        stats = self.test()  
        stats_train = self.train_error_and_loss()
        
        # Global Metrics (Weighted Average)
        glob_acc = np.sum(stats[2])*1.0/np.sum(stats[1])
        # stats[3] is list of average losses per client, stats[1] is list of num samples
        # Weighted average loss = sum(avg_loss * num_samples) / sum(num_samples)
        test_loss = sum([x * y for (x, y) in zip(stats[3], stats[1])]) / np.sum(stats[1])
        
        train_acc = np.sum(stats_train[2])*1.0/np.sum(stats_train[1])
        # train_loss = np.dot(stats_train[3], stats_train[1])*1.0/np.sum(stats_train[1])
        train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]).item() / np.sum(stats_train[1])
        
        self.rs_glob_acc.append(glob_acc)
        self.rs_train_acc.append(train_acc)
        self.rs_train_loss.append(train_loss)
        
        print("Average Global Accurancy: ", glob_acc)
        print("Average Global Trainning Accurancy: ", train_acc)
        print("Average Global Trainning Loss: ",train_loss)
        print("Average Global Test Loss: ", test_loss)

        # Client-wise Metrics (Distribution)
        # stats[3] is AVERAGE loss per client (from User.test), stats[1] is num samples per client
        client_test_losses = stats[3]
        client_test_accs = [c / n for c, n in zip(stats[2], stats[1])]
        
        # stats_train[3] is mean loss per client (tensor or float)
        client_train_losses = [l.item() if isinstance(l, torch.Tensor) else l for l in stats_train[3]]
        client_train_accs = [c / n for c, n in zip(stats_train[2], stats_train[1])]

        loss_stats = self.calculate_stats(client_train_losses)
        acc_stats = self.calculate_stats(client_train_accs)
        test_loss_stats = self.calculate_stats(client_test_losses)
        test_acc_stats = self.calculate_stats(client_test_accs)

        # Traffic Estimation
        model_size = self.get_model_size()
        num_participants = self.num_users # This is users per round
        # Assuming 1 round of communication (Server -> Client -> Server)
        # Tx: Server sends global model to clients
        # Rx: Clients send updates to server
        tx_bytes = model_size * num_participants
        rx_bytes = model_size * num_participants
        
        avg_tx_bytes = tx_bytes / max(num_participants, 1)
        avg_rx_bytes = rx_bytes / max(num_participants, 1)
        avg_tx_mb = avg_tx_bytes / (1024 * 1024)
        avg_rx_mb = avg_rx_bytes / (1024 * 1024)
        
        # Determine log keys based on algorithm
        # For pFedMe, evaluate() is for the global model, which we distinguish from the personalized model
        prefix = ""
        # if self.algorithm in ["pFedMe", "pFedMe_p"]:
        #     prefix = "global/"

        log_data = build_round_log(
            round_idx=(round_idx + 1) if round_idx is not None else 0,
            meta_loss=None,
            client_losses=client_train_losses,
            client_accs=client_train_accs,
            val_losses=client_test_losses,
            val_accs=client_test_accs,
            duration=duration,
            tx_bytes=avg_tx_bytes,
            rx_bytes=avg_rx_bytes,
            extra_metrics=extra_metrics,
        )
        wandb.log(log_data, step=round_idx, commit=commit)

        # Append per-round convergence metrics to CSV under algorithm-specific directory
        try:
            algo_dir = "perfedavg" if self.algorithm == "PerAvg" else ("pfedme" if self.algorithm in ["pFedMe", "pFedMe_p"] else self.algorithm.lower())
            base_dir = getattr(self, 'save_dir', 'save')
            out_dir = Path(base_dir) / algo_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            csv_path = out_dir / "convergence.csv"
            row = {
                "round": int((round_idx + 1) if round_idx is not None else 0),
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "test_loss": float(test_loss),
                "test_acc": float(glob_acc),
                "max_client_duration": float(extra_metrics.get("max_client_duration")) if isinstance(extra_metrics, dict) and isinstance(extra_metrics.get("max_client_duration"), (int, float)) else None,
            }
            df = pd.DataFrame([row])
            header = not csv_path.exists()
            df.to_csv(csv_path, mode="a", header=header, index=False)
        except Exception as e:
            print(f"[ServerBase] Failed to write convergence.csv: {e}")

    def evaluate_personalized_model(self, round_idx=None, commit=True, *, extra_metrics=None, duration=None):
        stats = self.test_persionalized_model()  
        stats_train = self.train_error_and_loss_persionalized_model()
        glob_acc = np.sum(stats[2])*1.0/np.sum(stats[1])
        # stats[3] is list of average losses per client, stats[1] is list of num samples
        test_loss = sum([x * y for (x, y) in zip(stats[3], stats[1])]) / np.sum(stats[1])
        
        train_acc = np.sum(stats_train[2])*1.0/np.sum(stats_train[1])
        # train_loss = np.dot(stats_train[3], stats_train[1])*1.0/np.sum(stats_train[1])
        train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]).item() / np.sum(stats_train[1])
        self.rs_glob_acc_per.append(glob_acc)
        self.rs_train_acc_per.append(train_acc)
        self.rs_train_loss_per.append(train_loss)
        #print("stats_train[1]",stats_train[3][0])
        print("Average Personal Accurancy: ", glob_acc)
        print("Average Personal Trainning Accurancy: ", train_acc)
        print("Average Personal Trainning Loss: ",train_loss)
        print("Average Personal Test Loss: ", test_loss)
        
        # Client-wise Metrics (Distribution)
        client_test_losses = stats[3]
        client_test_accs = [c / n for c, n in zip(stats[2], stats[1])]
        client_train_losses = [l.item() if isinstance(l, torch.Tensor) else l for l in stats_train[3]]
        client_train_accs = [c / n for c, n in zip(stats_train[2], stats_train[1])]

        loss_stats = self.calculate_stats(client_train_losses)
        acc_stats = self.calculate_stats(client_train_accs)
        test_loss_stats = self.calculate_stats(client_test_losses)
        test_acc_stats = self.calculate_stats(client_test_accs)

        # Traffic (Same as global for now, or could be different if pFedMe does extra steps)
        # pFedMe does K steps of local updates, but communication is still 1 round per global iter usually.
        model_size = self.get_model_size()
        num_participants = self.num_users
        tx_bytes = model_size * num_participants
        rx_bytes = model_size * num_participants
        avg_tx_bytes = tx_bytes / max(num_participants, 1)
        avg_rx_bytes = rx_bytes / max(num_participants, 1)
        avg_tx_mb = avg_tx_bytes / (1024 * 1024)
        avg_rx_mb = avg_rx_bytes / (1024 * 1024)

        log_data = build_round_log(
            round_idx=(round_idx + 1) if round_idx is not None else 0,
            meta_loss=None,
            client_losses=client_train_losses,
            client_accs=client_train_accs,
            val_losses=client_test_losses,
            val_accs=client_test_accs,
            duration=duration,
            tx_bytes=avg_tx_bytes,
            rx_bytes=avg_rx_bytes,
            extra_metrics=extra_metrics,
        )
        wandb.log(log_data, step=round_idx, commit=commit)

    def evaluate_one_step(self):
        for c in self.users:
            c.train_one_step()

        stats = self.test()  
        stats_train = self.train_error_and_loss()

        # set local model back to client for training process.
        for c in self.users:
            c.update_parameters(c.local_model)

        glob_acc = np.sum(stats[2])*1.0/np.sum(stats[1])
        train_acc = np.sum(stats_train[2])*1.0/np.sum(stats_train[1])
        # train_loss = np.dot(stats_train[3], stats_train[1])*1.0/np.sum(stats_train[1])
        train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]).item() / np.sum(stats_train[1])
        self.rs_glob_acc_per.append(glob_acc)
        self.rs_train_acc_per.append(train_acc)
        self.rs_train_loss_per.append(train_loss)
        #print("stats_train[1]",stats_train[3][0])
        print("Average Personal Accurancy: ", glob_acc)
        print("Average Personal Trainning Accurancy: ", train_acc)
        print("Average Personal Trainning Loss: ",train_loss)
