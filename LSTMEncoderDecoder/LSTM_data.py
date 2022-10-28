import os
import torch
from torch import device
import glob
import datetime
import numpy as np
from pathlib import Path
import pickle

class LSTMDataset(Dataset):
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
    
   ef __len__(self):
        return len(self.df) - self.seq_len -1
    
    def __getitem__(self, idx):
        return (torch.tensor(self.sequence[idx], dtype = torch.float), torch.tensor(self.sequence[idx+1], dtype = torch.float))