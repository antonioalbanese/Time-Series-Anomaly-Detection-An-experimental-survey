import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torchinfo import summary
import torch.nn as nn
from torch.utils.data import DataLoader

from DeepAnt.deepant import TrafficDataset, DeepAnt, AnomalyDetector, DataModule

class Solver(object):
    DEFAULTS = {}

    def __init__(self, config, data_path):
        self.config = config
        self.data_path = data_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.prepare_data(data_path)
        self.build_model()
        

    def prepare_data(self, data_path):
        df = pd.read_csv(self.data_path, index_col = 'timestamp', parse_dates=['timestamp'])
        self.dataset = TrafficDataset(df, self.config['SEQ_LEN'])
        self.train_dl = DataLoader(self.dataset, batch_size = 32, num_workers = 10, pin_memory = True, shuffle = True)
        self.test_dl = DataLoader(self.dataset, batch_size = 1, num_workers = 10, pin_memory = True, shuffle = False)

    def build_model(self):
        self.model = DeepAnt(self.config['SEQ_LEN'], self.config['out_dim'])
        self.anomaly_detector = AnomalyDetector(self.model)
        self.criterion = nn.L1Loss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 1e-5)

    def train(self):
        loss_list = []
        self.model.train()
        self.model.to(self.device)
        for epoch in range(self.config["EPOCHS"]):
            curr_loss = 0
            for batch in self.train_dl:
                input = batch[0].to(self.device)
                output = self.model(input)
                loss = self.criterion(output, input)
                curr_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            loss_list.append(curr_loss)
        print(f"Epoch {epoch}: loss:{curr_loss}")
        return loss_list
    
    def test(self):
        output = self.trainer.predict(self.anomaly_detector, self.dm)
        o = output.copy()
        return o
