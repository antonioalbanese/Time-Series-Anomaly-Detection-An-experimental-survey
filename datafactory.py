import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import MinMaxScaler

class MyDataset(Dataset):
    def __init__(self, dataset: str, data_path: str, seq_len: int, step: int, method: str, mode: str, hidden_size: int):
        if dataset == 'NAB' and (data_path == "" or data_path == None):
            sys.exit('When dataset is NAB, data_path must have a path to the dataset to be used')
        if method == "USA" and hidden_size == None:
            sys.exit('USAD method need HIDDEN_SIZE as hyperparameter')
        
        self.seq_len = seq_len
        self.dataset = dataset
        self.data_path = data_path
        self.method = method
        self.mode = mode
        self.step = step
        self.hidden_size = hidden_size



        self.load_data()


    def load_data(self):
        scaler = MinMaxScaler()
        window_size = self.seq_len
        step = self.step
        
        if self.dataset == 'SWAT':
            """SWAT DATASET MUST BE DOWNLOADED IN data/SWAT"""
            if self.mode == "TRAIN":
                #load the training data
                data = pd.read_csv("data/SWAT/SWaT_Dataset_Normal_v1.csv")
                data = data.drop(["Timestamp" , "Normal/Attack" ] , axis = 1)
                #convert data to float
                for i in list(data): 
                    data[i]=data[i].apply(lambda x: str(x).replace("," , "."))
                data = data.astype(float)
                #scale data
                data = pd.DataFrame(scaler.fit_transform(data.values))
                #window train data
                limit = int(len(data))
                self.sequences = data.values[np.arange(window_size)[None, :] + np.arange(0,limit-window_size, step)[:, None]]
                self.n_sequences = self.sequences.shape[0]
                self.n_features = data.values.shape[-1]
            elif self.mode == 'TEST':
                ### load the test data
                data = pd.read_csv("data/SWAT/SWaT_Dataset_Attack_v0.csv",sep=";")
                data = data.drop(["Timestamp" , "Normal/Attack" ] , axis = 1)
                #convert data to float
                for i in list(data):
                    data[i]=data[i].apply(lambda x: str(x).replace("," , "."))
                data = data.astype(float)
                data = pd.DataFrame(scaler.fit_transform(data.values))
                ### window test data
                # limit = int(len(data))
                # self.sequences = data.values[np.arange(window_size)[None, :] + np.arange(0,limit-window_size, step)[:, None]]
                # # self.sequences = data.values[np.arange(window_size)[None, :] + np.arange(0,data.shape[0]-window_size, step)[:, None]] ##[n_sequences, window_size, features]
                # self.n_sequences = self.sequences.shape[0]
                # self.n_features = data.values.shape[-1]
                self.data = data.values
                self.n_sequences = self.data.shape//self.step +1
                self.n_features = self.data.shape[-1]
            ### create labels if method is DeepAnt    
            if self.method == "DEEPANT":
                self.labels = data.values[np.arange(step,data.shape[0]-1, step)[:, None]]
        elif self.dataset == 'MSL':
            """MSL DATASET MUST BE DOWNLOADED IN data/MSL
            DATAPATH is not needed"""
            if self.mode == "TRAIN":
                #load the training data
                data = np.load("./data/MSL/MSL_train.npy")
                #scale data
                data = pd.DataFrame(scaler.fit_transform(data))
                #window train data
                self.sequences = data.values[np.arange(window_size)[None, :] + np.arange(0,data.shape[0]-window_size, step)[:, None]]
                self.n_sequences = self.sequences.shape[0]
                self.n_features = data.values.shape[-1]
            elif self.mode == 'TEST':
                data = np.load("./data/MSL/MSL_test.npy")
                data = pd.DataFrame(scaler.fit_transform(data))
                ### window test data
                self.sequences = data.values[np.arange(window_size)[None, :] + np.arange(0,data.shape[0]-window_size, step)[:, None]]
                self.n_sequences = self.sequences.shape[0]
                self.n_features = data.values.shape[-1]
            ### create labels if method is DeepAnt    
            if self.method == "DEEPANT":
                self.labels = data.values[np.arange(step,data.shape[0]-1, step)[:, None]]
        elif self.dataset == 'SMD':
            """SMD DATASET MUST BE DOWNLOADED IN data/MSL
            DATAPATH is not needed"""
            if self.mode == "TRAIN":
                #load the training data
                data = np.load("./data/SMD/SMD_train.npy")
                #scale data
                data = pd.DataFrame(scaler.fit_transform(data))
                #window train data
                self.sequences = data.values[np.arange(window_size)[None, :] + np.arange(0,data.shape[0]-window_size, step)[:, None]]
                self.n_sequences = self.sequences.shape[0]
                self.n_features = data.values.shape[-1]
            elif self.mode == 'TEST':
                data = np.load("./data/SMD/SMD_test.npy")
                data = pd.DataFrame(scaler.fit_transform(data))
                ### window test data
                self.sequences = data.values[np.arange(window_size)[None, :] + np.arange(0,data.shape[0]-window_size, step)[:, None]]
                self.n_sequences = self.sequences.shape[0]
                self.n_features = data.values.shape[-1]
            ### create labels if method is DeepAnt    
            if self.method == "DEEPANT":
                self.labels = data.values[np.arange(step,data.shape[0]-1, step)[:, None]]
        elif self.dataset == 'NAB':
            print("NAB DS")
            """
            NAB DATASET MUST BE IN ./NAB/*datasets*/*file.csv*
            data_path MUST CONTAIN *datasets*/*file.csv* ONLY PART TO LOAD LABELS LATER
            """
            ### ! mode IS NOT NEEDED FOR NUMENTA AS WE WILL USE THE SAME DATASET FOR TRAINING AND TEST
            data = pd.read_csv("./NAB/" + self.data_path)
            data = data.drop(["timestamp"] , axis = 1)
            data = pd.DataFrame(scaler.fit_transform(data.values))
            data = pd.DataFrame(scaler.fit_transform(data.values))
            self.sequences = data.values[np.arange(window_size)[None, :] + np.arange(0,data.shape[0]-window_size, step)[:, None]]
            self.n_sequences = self.sequences.shape[0]
            self.n_features = data.values.shape[-1]
            ### create labels if method is DeepAnt  
            if self.method == "DEEPANT":
                self.labels = data.values[np.arange(step,data.shape[0]-1, step)[:, None]]

        if self.method == 'USAD':
            self.w_size = self.n_features*self.seq_len #shape[2]*shape[1]
            self.z_size = self.seq_len*self.hidden_size
    
    def get_sizes(self):
        return self.w_size, self.z_size

    def __len__(self):
        return self.n_sequences
    
    def __getitem__(self, idx):
        if self.method == 'DEEPANT':
            return (torch.tensor(self.sequences[idx], dtype=torch.float),
                    torch.tensor(self.labels[idx].reshape(self.labels[idx].shape[-1]), dtype = torch.float))
        elif self.method == 'USAD':
            # return torch.tensor(self.sequences[idx], dtype = torch.float).view(([self.w_size]))
            i = idx*self.step
            return torch.tensor(self.data[i:self.seq_len], dtype=torch.float).reshape(self.w_size)
        else: 
            return torch.tensor(self.sequences[idx], dtype = torch.float)