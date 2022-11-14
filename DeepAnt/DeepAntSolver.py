import pandas as pd
# import pytorch_lightning as pl
import torch
# from pytorch_lightning.callbacks import ModelCheckpoint
# from torchinfo import summary
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from DeepAnt.deepant import TrafficDataset, DeepAnt, MSLDataset
from sklearn.metrics import classification_report
import sklearn
import json

class DeepAntSolver(object):
    DEFAULTS = {}

    def __init__(self, config, data_path):
        self.config = config
        self.data_path = data_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.prepare_data(data_path)
        self.build_model()
        

    def prepare_data(self, data_path):
        if self.config['DATASET'] == 'NAB':
            self.n_feat = 1
            df = pd.read_csv(self.data_path, index_col = 'timestamp', parse_dates=['timestamp'])
            self.dataset = TrafficDataset(df, self.config['SEQ_LEN'])
            self.train_dl = DataLoader(self.dataset, batch_size = 32, num_workers = 10, pin_memory = True, shuffle = False)
            self.test_dl = DataLoader(self.dataset, batch_size = 1, num_workers = 10, pin_memory = True, shuffle = False)
        elif self.config['DATASET'] == 'MSL':
            self.n_feat = 55
            data = np.load(self.data_path + "/MSL_train.npy")
            test_data = np.load(self.data_path + "/MSL_test.npy")
            df = pd.DataFrame(data)
            test_df = pd.DataFrame(test_data)
            self.dataset = MSLDataset(df, self.config['SEQ_LEN'])
            self.test_dataset = MSLDataset(test_df, self.config['SEQ_LEN'])
            self.train_dl = DataLoader(self.dataset, batch_size = 32, num_workers = 10, pin_memory = True, shuffle = False)
            self.test_dl = DataLoader(self.test_dataset, batch_size = 1, num_workers = 10, pin_memory = True, shuffle = False)
        elif self.config['DATASET'] == 'SWAT':
            self.n_feat = 51
            ### load train data
            train_data = pd.read_csv("input/SWaT_Dataset_Normal_v1.csv", index_col = 'Timestamp', parse_dates=['timestamp'])
            train_data = train_data.drop(["Normal/Attack" ] , axis = 1)
            for i in list(train_data): 
                train_data[i]=train_data[i].apply(lambda x: str(x).replace("," , "."))
            train_data = train_data.astype(float)
            ### train_data is df of float with timestamp as index
            self.dataset = SWATDataset(train_data, self.config['SEQ_LEN'])
            self.train_dl = DataLoader(self.dataset, batch_size = 32, num_workers = 10, pin_memory = True, shuffle = False)
            
            ### load test data
            attack = pd.read_csv("input/SWaT_Dataset_Attack_v0.csv",sep=";", index_col = 'Timestamp', parse_dates=['timestamp'])
            attack = attack.drop(["Normal/Attack"] , axis = 1)
            for i in list(attack):
                attack[i]=attack[i].apply(lambda x: str(x).replace("," , "."))
            attack = attack.astype(float)
            self.test_dataset = SWATDataset(attack, self.config['SEQ_LEN'])
            self.test_dl = DataLoader(self.test_dataset, batch_size = 1, num_workers = 10, pin_memory = True, shuffle = False)

    def build_model(self):
        self.model = DeepAnt(self.n_feat, self.config['SEQ_LEN'], 55)
        # self.anomaly_detector = AnomalyDetector(self.model)
        self.criterion = nn.L1Loss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.config['LR'])

    def train(self):
        loss_list = []
        self.model.train()
        self.model.to(self.device)
        for epoch in range(self.config["EPOCHS"]):
            curr_loss = 0
            for i,batch in enumerate(self.train_dl):
                input, y = batch
                input = input.to(self.device)
                y = y.to(self.device)
                output = self.model(input)
                print(output.size())
                loss = self.criterion(output, y)
                curr_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            loss_list.append(curr_loss/len(self.train_dl))
            if self.config['VERBOSE']:
                print(f"Epoch {epoch}: loss:{curr_loss}")
        return loss_list
    
    def test(self):
        with torch.no_grad():
            self.model.eval()
            loss_list = []
            predictions = []
            for i, batch in enumerate(self.test_dl):
                input, y = batch
                input = input.to(self.device)
                y = y.to(self.device)
                output = self.model(input)
                loss = torch.linalg.norm(output-y)
                predictions.append(output.cpu().numpy())
                loss_list.append(loss.detach().item())
            
            sc = sklearn.preprocessing.MinMaxScaler(feature_range=(0,1))
            losses = sc.fit_transform(np.array(loss_list).reshape(-1, 1))
            true_labels = self.load_true_labels(self.data_path, losses.shape[0])
            reports = []
            
            # if self.config['TH_SEARCH']:
            #     for th in self.config['TH_LIST']:
            #         bool_losses = losses.copy()
            #         for i,el in enumerate(losses):
            #                 if el >= th:
            #                     bool_losses[i] = 1
            #                 else:
            #                     bool_losses[i] = 0
            #         report = classification_report(true_labels, bool_losses, output_dict=True)
            #         reports.append(report)
            #     return predictions, losses, reports
            # else: 
            bool_losses = losses.copy()
            for i,el in enumerate(losses):
                    if el >= self.config['CONFIDENCE']:
                        bool_losses[i] = 1
                    else:
                        bool_losses[i] = 0
            report = classification_report(true_labels, bool_losses, output_dict=True)
        return predictions, losses, report


    def load_true_labels(self, data_path, l):
        if self.config['DATASET'] == 'NAB':
            with open("./NAB/combined_windows.json") as FI:
                j_label = json.load(FI)
            key = data_path.split("./NAB/")[-1]
            windows = j_label[key]

            df = pd.read_csv(data_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            y = np.zeros(len(df))
            for w in windows:
                start = pd.to_datetime(w[0])
                end = pd.to_datetime(w[1])
                for idx in df.index:
                    if df.loc[idx, 'timestamp'] >= start and df.loc[idx, 'timestamp'] <= end:
                        y[idx] = 1.0
        elif self.config['DATASET'] == 'MSL':
            y = np.load(data_path + "/MSL_test_label.npy")
            
        return y[:l]