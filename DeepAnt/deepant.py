import torch
import torch.nn as nn
import numpy as np
import math
# import pytorch_lightning as pl

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset


class TrafficDataset(Dataset):
    def __init__(self, df, seq_len):
        self.df = df
        self.seq_len = seq_len
        self.sequence, self.labels, self.timestamp = self.create_sequence(df, seq_len)

    def create_sequence(self, df, seq_len):
        sc = MinMaxScaler()
        index = df.index.to_numpy()
        ts = sc.fit_transform(df.value.to_numpy().reshape(-1, 1))
        
        sequence = []
        label = []
        timestamp = []
        for i in range(len(ts) - seq_len):
            sequence.append(ts[i:i+seq_len])
            label.append(ts[i+seq_len])
            timestamp.append(index[i+seq_len])
            
            
        return np.array(sequence), np.array(label), np.array(timestamp)
    
    def __len__(self):
        return len(self.df) - self.seq_len
    
    def __getitem__(self, idx):
        return (torch.tensor(self.sequence[idx], dtype = torch.float).permute(1, 0), 
                torch.tensor(self.labels[idx], dtype = torch.float))

# class DataModule(pl.LightningDataModule):
#     def __init__(self, df, seq_len):
#         super().__init__()
#         self.df = df
#         self.seq_len = seq_len
#     def setup(self, stage=None):
#         self.dataset = TrafficDataset(self.df, self.seq_len)
        
#     def train_dataloader(self):
#         return DataLoader(self.dataset, batch_size = 32, num_workers = 10, pin_memory = True, shuffle = True)
    
#     def predict_dataloader(self):
#         return DataLoader(self.dataset, batch_size = 1, num_workers = 10, pin_memory = True, shuffle = False)


class DeepAnt(nn.Module):
    def __init__(self, seq_len, p_w):
        super().__init__()
        conv1_k_size = 3
        conv1_out = seq_len - conv1_k_size
        maxp1_k_size = 2
        maxp1_out = math.ceil((conv1_out - maxp1_k_size)/maxp1_k_size + 1)

        conv2_k_size = 3
        conv2_out = maxp1_out - conv2_k_size
        maxp2_k_size = 2
        maxp2_out = math.ceil((conv2_out - maxp2_k_size)/maxp2_k_size + 1)

        out_channels = 32
        flatten_out = out_channels*maxp2_out
        print(flatten_out)
        self.convblock1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=out_channels, kernel_size=conv1_k_size, padding='valid'),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=maxp1_k_size)
        )
        
        self.convblock2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=out_channels, kernel_size=conv2_k_size, padding='valid'), #11
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=maxp2_k_size) ## (11-2)/2 + 1 = 6
        )
        
        self.flatten = nn.Flatten() ## out_channels*dim_out_maxpool
        
        self.denseblock = nn.Sequential(
            nn.Linear(flatten_out, 40),
            #nn.Linear(96, 40), # for SEQL_LEN = 20
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
        )
        self.out = nn.Linear(flatten_out, p_w)
        
    def forward(self, x):
        # print("input")
        # print(x.size())
        # print('##################')
        x = self.convblock1(x)
        # print("conv1")
        # print(x.size())
        # print('##################')
        x = self.convblock2(x)
        # print("conv2")
        # print(x.size())
        # print('##################')
        x = self.flatten(x)
        # print("flatten")
        # print(x.size())
        # print('##################')
        x = self.denseblock(x)
        x = self.out(x)
        return x


# class AnomalyDetector(pl.LightningModule):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model
#         self.criterion = nn.L1Loss()
#     def forward(self, x):
#         return self.model(x)
    
#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         y_pred = self(x)
#         loss = self.criterion(y_pred, y)
#         self.log('train_loss', loss, prog_bar=True, logger = True)
#         return loss
#     def predict_step(self, batch, batch_idx):
#         x, y = batch
#         y_pred = self(x)
#         return y_pred, torch.linalg.norm(y_pred-y)
    
#     def configure_optimizers(self):
#         return torch.optim.Adam(self.parameters(), lr = 1e-5)
    