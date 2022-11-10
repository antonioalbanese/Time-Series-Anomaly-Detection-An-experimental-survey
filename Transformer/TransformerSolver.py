import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from Transformer.utils.utils import *
from Transformer.model.AnomalyTransformer import AnomalyTransformer
from Transformer.data_factory.data_loader import get_loader_segment
from sklearn.metrics import classification_report
import sklearn
from Transformer.transformer import my_kl_loss, adjust_learning_rate, EarlyStopping, load_data







class TransformerSolver(object):
    DEFAULTS = {}

    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.__dict__.update(Solver.DEFAULTS, **config)
        # self.lr = config['lr']
        # self.num_epochs = config['num_epochs']
        # self.k = config['k']
        # self.win_size = config['win_size']
        # self.input_c = config['input_c']
        # self.output_c = config['output_c']
        # self.batch_size = config['batch_size']
        # self.pretrained_model = config['pretrained_model']
        self.dataset = config['DATASET']
        # self.mode = config['mode']
        self.data_path = config['data_path']
        # self.model_save_path = config['model_save_path']
        # self.anormly_ratio = config['anormly_ratio']


        self.train_loader, self.vali_loader, self.test_loader, self.thre_loader = load_data(self.data_path, 
                                                                                            batch_size=self.config['BATCH_SIZE'], 
                                                                                            win_size=self.config['SEQ_LEN'],
                                                                                            dataset=self.dataset)

        self.build_model()
        
        self.criterion = nn.MSELoss()

    def build_model(self):
        self.model = AnomalyTransformer(win_size=self.config['SEQ_LEN'], 
                                        enc_in=self.config['INPUT_C'], 
                                        c_out=self.config['OUTPUT_C'], 
                                        e_layers=3).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['LR'])


    def vali(self, vali_loader):
        self.model.eval()

        loss_1 = []
        loss_2 = []
        for i, (input_data, _) in enumerate(vali_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                series_loss += (torch.mean(my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               self.win_size)).detach())) + torch.mean(
                    my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)).detach(),
                        series[u])))
                prior_loss += (torch.mean(
                    my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)),
                               series[u].detach())) + torch.mean(
                    my_kl_loss(series[u].detach(),
                               (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)))))
            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)

            rec_loss = self.criterion(output, input)
            loss_1.append((rec_loss - self.k * series_loss).item())
            loss_2.append((rec_loss + self.k * prior_loss).item())

        return np.average(loss_1), np.average(loss_2)

    def train(self):

        print("======================TRAIN MODE======================")

        time_now = time.time()
        # path = self.model_save_path
        # if not os.path.exists(path):
        #     os.makedirs(path)
        # early_stopping = EarlyStopping(patience=3, verbose=True, dataset_name=self.dataset)
        train_steps = len(self.train_loader)
        history = {
            'train_loss': [], 
            'val_loss1':[],
            'val_loss2':[]
        }

        for epoch in range(self.num_epochs):
            iter_count = 0
            loss1_list = []

            epoch_time = time.time()
            self.model.train()
            for i, (input_data, labels) in enumerate(self.train_loader):

                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)

                output, series, prior, _ = self.model(input)

                # calculate Association discrepancy
                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    a = torch.mean(my_kl_loss(series[u], (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.config['SEQ_LEN'])).detach()))
                    b = torch.mean(my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.config['SEQ_LEN'])).detach(), series[u]))
                    series_loss += (a + b)

                    prior_a = torch.mean(my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.config['SEQ_LEN'])), series[u].detach()))
                    prior_b = torch.mean(my_kl_loss(series[u].detach(), (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.config['SEQ_LEN']))))
                    prior_loss += (prior_a + prior_b)
                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)

                rec_loss = self.criterion(output, input)

                loss1_list.append((rec_loss - self.config['K'] * series_loss).item())
                loss1 = rec_loss - self.config['K'] * series_loss
                loss2 = rec_loss + self.config['K'] * prior_loss

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                # Minimax strategy
                loss1.backward(retain_graph=True)
                loss2.backward()
                self.optimizer.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(loss1_list)

            vali_loss1, vali_loss2 = self.vali(self.test_loader)

            history['train_loss'].append(train_loss.item())
            history['val_loss1'].append(vali_loss1.item())
            history['val_loss2'].append(vali_loss2.item())

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                    epoch + 1, train_steps, train_loss, vali_loss1))
            # early_stopping(vali_loss1, vali_loss2, self.model, path)
            # if early_stopping.early_stop:
            #     print("Early stopping")
            #     break
            adjust_learning_rate(self.optimizer, epoch + 1, self.config['LR'])
            
        return history

    def test(self):
        with torch.no_grad():
            # self.model.load_state_dict(
            #     torch.load(
            #         os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint.pth')))
            self.model.eval()
            temperature = 50

            print("======================TEST MODE======================")

            criterion = nn.MSELoss(reduce=False)

            # (1) stastic on the train set
            attens_energy = []
            for i, (input_data, labels) in enumerate(self.train_loader):
                input = input_data.float().to(self.device)
                output, series, prior, _ = self.model(input)
                loss = torch.mean(criterion(input, output), dim=-1)
                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    if u == 0:
                        series_loss = my_kl_loss(series[u], (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.config['SEQ_LEN'])).detach()) * temperature
                        prior_loss = my_kl_loss(
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.config['SEQ_LEN'])),
                            series[u].detach()) * temperature
                    else:
                        series_loss += my_kl_loss(series[u], (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.config['SEQ_LEN'])).detach()) * temperature
                        prior_loss += my_kl_loss(
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.config['SEQ_LEN'])),
                            series[u].detach()) * temperature

                metric = torch.softmax((-series_loss - prior_loss), dim=-1)
                cri = metric * loss
                cri = cri.detach().cpu().numpy()
                attens_energy.append(cri)

            attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
            train_energy = np.array(attens_energy)

            # (2) find the threshold
            attens_energy = []
            for i, (input_data, labels) in enumerate(self.thre_loader):
                input = input_data.float().to(self.device)
                output, series, prior, _ = self.model(input)

                loss = torch.mean(criterion(input, output), dim=-1)

                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    if u == 0:
                        series_loss = my_kl_loss(series[u], (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.config['SEQ_LEN'])).detach()) * temperature
                        prior_loss = my_kl_loss(
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.config['SEQ_LEN'])),
                            series[u].detach()) * temperature
                    else:
                        series_loss += my_kl_loss(series[u], (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.config['SEQ_LEN'])).detach()) * temperature
                        prior_loss += my_kl_loss(
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.config['SEQ_LEN'])),
                            series[u].detach()) * temperature
                # Metric
                metric = torch.softmax((-series_loss - prior_loss), dim=-1)
                cri = metric * loss
                cri = cri.detach().cpu().numpy()
                attens_energy.append(cri)

            attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
            test_energy = np.array(attens_energy)
            combined_energy = np.concatenate([train_energy, test_energy], axis=0)
            thresh = np.percentile(combined_energy, 100 - self.config['RATIO'])
            print("Threshold :", thresh)

            # (3) evaluation on the test set
            test_labels = []
            attens_energy = []
            for i, (input_data, labels) in enumerate(self.thre_loader):
                input = input_data.float().to(self.device)
                output, series, prior, _ = self.model(input)

                loss = torch.mean(criterion(input, output), dim=-1)

                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    if u == 0:
                        series_loss = my_kl_loss(series[u], (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.config['SEQ_LEN'])).detach()) * temperature
                        prior_loss = my_kl_loss(
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.config['SEQ_LEN'])),
                            series[u].detach()) * temperature
                    else:
                        series_loss += my_kl_loss(series[u], (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.config['SEQ_LEN'])).detach()) * temperature
                        prior_loss += my_kl_loss(
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.config['SEQ_LEN'])),
                            series[u].detach()) * temperature
                metric = torch.softmax((-series_loss - prior_loss), dim=-1)

                cri = metric * loss
                cri = cri.detach().cpu().numpy()
                attens_energy.append(cri)
                test_labels.append(labels)

            attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
            test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
            test_energy = np.array(attens_energy)
            test_labels = np.array(test_labels)

            pred = (test_energy > thresh).astype(int)

            gt = test_labels.astype(int)

            # print("pred:   ", pred.shape)
            # print("gt:     ", gt.shape)

            # detection adjustment
            anomaly_state = False
            for i in range(len(gt)):
                if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                    anomaly_state = True
                    for j in range(i, 0, -1):
                        if gt[j] == 0:
                            break
                        else:
                            if pred[j] == 0:
                                pred[j] = 1
                    for j in range(i, len(gt)):
                        if gt[j] == 0:
                            break
                        else:
                            if pred[j] == 0:
                                pred[j] = 1
                elif gt[i] == 0:
                    anomaly_state = False
                if anomaly_state:
                    pred[i] = 1

            pred = np.array(pred)
            gt = np.array(gt)
            # print("pred: ", pred.shape)
            # print("gt:   ", gt.shape)

            # from sklearn.metrics import precision_recall_fscore_support
            # from sklearn.metrics import accuracy_score
            # accuracy = accuracy_score(gt, pred)
            # precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
            #                                                                       average='binary')
            # print(
            #     "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            #         accuracy, precision,
            #         recall, f_score))

            report = classification_report(gt, pred, output_dict=True)

        return  test_energy, report
