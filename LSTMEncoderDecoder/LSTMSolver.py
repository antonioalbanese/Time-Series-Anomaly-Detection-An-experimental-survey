import pandas as pd
import torch
from torchinfo import summary
import torch.nn as nn
from torch.utils.data import DataLoader
from LSTMEncoderDecoder.LSTMmodel import RNNPredictor
from LSTMEncoderDecoder.LSTM_data import LSTMDataset
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
        self.dataset = LSTMDataset(df, self.config['SEQ_LEN'])
        self.train_dl = DataLoader(self.dataset, batch_size = 32, num_workers = 10, pin_memory = True, shuffle = True)
        self.test_dl = DataLoader(self.dataset, batch_size = 1, num_workers = 10, pin_memory = True, shuffle = False)

    def build_model(self):
        self.model = RNNPredictor('LSTM', 
                                   self.config['IN_FEAT'],
                                   self.config['EMSIZE'],
                                   self.config['NHID'],
                                   self.config['IN_FEAT'], #dec_out_size is in_features
                                   self.config['NLAYERS'],
                                   self.config['DROPOUT'],
                                   self.config['TIED'],
                                   self.config['RES_CONNECTION']).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                          lr = self.config['LR'], 
                                          weight_decay=self.config['WEIGHT_DECAY'])
        


    def train(self):
        with torch.enable_grad():
            # Turn on training mode which enables dropout.
            self.model.train()
            total_loss = 0
            hidden = self.model.init_hidden(self.config['BATCH_SIZE'])
            for k, batch in enumerate(self.train_dl):
                inputSeq = batch[0].permute(1,0,-1).to(self.device)
                targetSeq = batch[1].permute(1,0,-1).to(self.device)

                # Starting each batch, we detach the hidden state from how it was previously produced.
                # If we didn't, the model would try backpropagating all the way to start of the dataset.
                hidden = self.model.repackage_hidden(hidden)
                hidden_ = self.model.repackage_hidden(hidden)
                self.optimizer.zero_grad()

                '''Loss1: Free running loss'''
                outVal = inputSeq[0].unsqueeze(0)
                outVals=[]
                hids1 = []
                for i in range(inputSeq.size(0)):
                    outVal, hidden_, hid = self.model.forward(outVal, hidden_,return_hiddens=True)
                    outVals.append(outVal)
                    hids1.append(hid)
                outSeq1 = torch.cat(outVals,dim=0)
                hids1 = torch.cat(hids1,dim=0)
                # loss1 = self.criterion(outSeq1.view(self.config['BATCH_SIZE'],-1), targetSeq.view(self.config['BATCH_SIZE'],-1))
                loss1 = self.criterion(outSeq1, targetSeq)

                '''Loss2: Teacher forcing loss'''
                outSeq2, hidden, hids2 = self.model.forward(inputSeq, hidden, return_hiddens=True)
                # loss2 = self.criterion(outSeq2.view(self.config['BATCH_SIZE'], -1), targetSeq.view(self.config['BATCH_SIZE'], -1))
                loss2 = self.criterion(outSeq2, targetSeq)

                '''Loss3: Simplified Professor forcing loss'''
                # loss3 = self.criterion(hids1.view(self.config['BATCH_SIZE'],-1), hids2.view(self.config['BATCH_SIZE'],-1).detach())
                loss3 = self.criterion(hids1, hids2.detach())

                '''Total loss = Loss1+Loss2+Loss3'''
                loss = loss1+loss2+loss3
                loss.backward()

                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['CLIP'])
                self.optimizer.step()

                total_loss += loss.item()

                if batch % self.config['LOG_INTERVAL'] == 0 and batch > 0:
                    cur_loss = total_loss / self.config['LOG_INTERVAL']
                    # elapsed = time.time() - start_time
                    print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.4f} | '
                        'loss {:5.2f} '.format(
                        epoch, batch, len(train_dataset) // self.config['SEQ_LEN'],
                                    elapsed * 1000 / self.config['LOG_INTERVAL'], cur_loss))
                    total_loss = 0
                    # start_time = time.time()
    
    def test(self):
        self.model.eval()
        loss_list = []
        predictions = []
        for i, batch in enumerate(self.test_dl):
            input, y = batch
            input = input.to(self.device)
            y = y.to(self.device)
            output = self.model(input)
            predictions.append(output.item())
            loss = torch.linalg.norm(output-y)
            loss_list.append(loss.detach().item())
        return predictions, loss_list