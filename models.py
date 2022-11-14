import torch
import torch.nn as nn
import numpy as np
# import pytorch_lightning as pl
import math

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset


class DeepAnt(nn.Module):
    def __init__(self, n_features, seq_len):
        super().__init__()
        block1_out = math.ceil(((seq_len-3)- 2)/2 + 1)
        block2_out = math.ceil((block1_out - 3 - 2)/2 + 1)
        flatten_out =  32 * block2_out
        
        self.convblock1 = nn.Sequential(
            nn.Conv1d(in_channels=n_features, out_channels=32, kernel_size=3, padding='valid'),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2)
        )
        
        self.convblock2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding='valid'),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2)
        )
        
        self.flatten = nn.Flatten()
        
        self.denseblock = nn.Sequential(
            nn.Linear(flatten_out, 40),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
        )
        self.out = nn.Linear(40, n_features)
        
    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.flatten(x)
        x = self.denseblock(x)
        x = self.out(x)
        return x