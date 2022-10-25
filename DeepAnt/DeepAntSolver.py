import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torchinfo import summary

from DeepAnt.deepant import TrafficDataset, DeepAnt, AnomalyDetector, DataModule

class Solver(object):
    DEFAULTS = {}

    def __init__(self, config, data_path):
        self.config = config
        self.data_path = data_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dm = self.prepare_data(data_path)
        self.build_model()
        

    def prepare_data(self, data_path):
        df = pd.read_csv(self.data_path, index_col = 'timestamp', parse_dates=['timestamp'])
        self. dataset = TrafficDataset(df, self.config['SEQ_LEN'])
        dm = DataModule(df, self.config['SEQ_LEN'])
        return dm

    def build_model(self):
        self.model = DeepAnt(self.config['SEQ_LEN'], self.config['out_dim'])
        self.anomaly_detector = AnomalyDetector(self.model)
        self.mc = ModelCheckpoint(
            dirpath = 'checkpoints',
            save_last = True,
            save_top_k = 1,
            verbose = True,
            monitor = 'train_loss', 
            mode = 'min'
            )
        self.mc.CHECKPOINT_NAME_LAST = f'DeepAnt-best-checkpoint'
        self.trainer = pl.Trainer(max_epochs=self.config['EPOCHS'], accelerator = self.device.type, callbacks = self.mc)

    def train(self):
        self.trainer.fit(self.anomaly_detector, self.dm)
    
    def test(self):
        output = self.trainer.predict(self.anomaly_detector, self.dm)
        print(output)
        ## TODO: look at what it does return
        return output.copy()
