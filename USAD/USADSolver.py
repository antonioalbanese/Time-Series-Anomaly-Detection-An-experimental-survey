import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from sklearn import preprocessing
import torch.utils.data as data_utils
from USAD.usad import *



class USADSolver(object):
    DEFAULTS = {}

    def __init__(self, config):
        self.config = config
        self.prepare_data()
        self.build_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()
    
    def prepare_data(self):
        ### load train data
        train_data = pd.read_csv("input/SWaT_Dataset_Normal_v1.csv")
        train_data = train_data.drop(["Timestamp" , "Normal/Attack" ] , axis = 1)
        for i in list(train_data): 
            train_data[i]=train_data[i].apply(lambda x: str(x).replace("," , "."))
        train_data = train_data.astype(float)
        min_max_scaler = preprocessing.MinMaxScaler()
        x = train_data.values
        x_scaled = min_max_scaler.fit_transform(x)
        train_data = pd.DataFrame(x_scaled)

        ### load test data
        attack = pd.read_csv("input/SWaT_Dataset_Attack_v0.csv",sep=";")#, nrows=1000)
        labels = [ float(label!= 'Normal' ) for label  in attack["Normal/Attack"].values]
        attack = attack.drop(["Timestamp" , "Normal/Attack" ] , axis = 1)
        for i in list(attack):
            attack[i]=attack[i].apply(lambda x: str(x).replace("," , "."))
        attack = attack.astype(float)
        x = attack.values 
        x_scaled = min_max_scaler.transform(x)
        attack = pd.DataFrame(x_scaled)

        ### window data
        window_size = self.config['SEQ_LEN']
        windows_normal=train_data.values[np.arange(window_size)[None, :] + np.arange(train_data.shape[0]-window_size)[:, None]]
        windows_attack=attack.values[np.arange(window_size)[None, :] + np.arange(attack.shape[0]-window_size)[:, None]]

        ### make dataloader
        BATCH_SIZE = self.config['BATCH_SIZE']
        hidden_size = self.config['HIDDEN_SIZE']
        w_size=windows_normal.shape[1]*windows_normal.shape[2]
        z_size=windows_normal.shape[1]*hidden_size
        self.w_size = w_size
        self.z_size = z_size

        windows_normal_train = windows_normal[:int(np.floor(.8 *  windows_normal.shape[0]))]
        windows_normal_val = windows_normal[int(np.floor(.8 *  windows_normal.shape[0])):int(np.floor(windows_normal.shape[0]))]

        self.train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
            torch.from_numpy(windows_normal_train).float().view(([windows_normal_train.shape[0],w_size]))
        ) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        self.val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
            torch.from_numpy(windows_normal_val).float().view(([windows_normal_val.shape[0],w_size]))
        ) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        self.test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
            torch.from_numpy(windows_attack).float().view(([windows_attack.shape[0],w_size]))
        ) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)




    def build_model(self):
        self.model = UsadModel(self.w_size, self.z_size).to(self.device)

    def train(self):
        history = []
        optimizer1 = opt_func(list(self.model.encoder.parameters())+list(self.model.decoder1.parameters()))
        optimizer2 = opt_func(list(self.model.encoder.parameters())+list(self.model.decoder2.parameters()))
        for epoch in range(self.config['EPOCHS']):
            for [batch] in self.train_loader:
                batch= batch.to(self.device)
                
                #Train AE1
                loss1,loss2 = self.model.training_step(batch,epoch+1)
                loss1.backward()
                optimizer1.step()
                optimizer1.zero_grad()
                
                
                #Train AE2
                loss1,loss2 = self.model.training_step(batch,epoch+1)
                loss2.backward()
                optimizer2.step()
                optimizer2.zero_grad()
                
                
            result = evaluate(epoch+1)
            model.epoch_end(epoch, result)
            history.append(result)
        return history
    
    def evaluate(n):
        outputs = [self.model.validation_step(batch.to(self.device), n) for [batch] in self.val_loader]
        return self.model.validation_epoch_end(outputs)
    

    def test(alpha=.5, beta=.5):
        results=[]
        for [batch] in self.test_loader:
            batch= batch.to(self.device)
            w1=self.model.decoder1(self.model.encoder(batch))
            w2=self.model.decoder2(self.model.encoder(w1))
            results.append(alpha*torch.mean((batch-w1)**2,axis=1)+beta*torch.mean((batch-w2)**2,axis=1))
        return results