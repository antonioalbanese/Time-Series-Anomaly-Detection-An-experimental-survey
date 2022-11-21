import torch
import torch.nn as nn
import numpy as np
# import pytorch_lightning as pl
import math

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

#### DEEPANT
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


#### USAD
class UsadEncoder(nn.Module):
  def __init__(self, in_size, latent_size):
    super().__init__()
    self.linear1 = nn.Linear(in_size, int(in_size/2))
    self.linear2 = nn.Linear(int(in_size/2), int(in_size/4))
    self.linear3 = nn.Linear(int(in_size/4), latent_size)
    self.relu = nn.ReLU(True)
        
  def forward(self, w):
    out = self.linear1(w)
    out = self.relu(out)
    out = self.linear2(out)
    out = self.relu(out)
    out = self.linear3(out)
    z = self.relu(out)
    return z
    
class UsadDecoder(nn.Module):
  def __init__(self, latent_size, out_size):
    super().__init__()
    self.linear1 = nn.Linear(latent_size, int(out_size/4))
    self.linear2 = nn.Linear(int(out_size/4), int(out_size/2))
    self.linear3 = nn.Linear(int(out_size/2), out_size)
    self.relu = nn.ReLU(True)
    self.sigmoid = nn.Sigmoid()
        
  def forward(self, z):
    out = self.linear1(z)
    out = self.relu(out)
    out = self.linear2(out)
    out = self.relu(out)
    out = self.linear3(out)
    w = self.sigmoid(out)
    return w
    
class UsadModel(nn.Module):
  def __init__(self, w_size, z_size):
    super().__init__()
    self.encoder = UsadEncoder(w_size, z_size)
    self.decoder1 = UsadDecoder(z_size, w_size)
    self.decoder2 = UsadDecoder(z_size, w_size)