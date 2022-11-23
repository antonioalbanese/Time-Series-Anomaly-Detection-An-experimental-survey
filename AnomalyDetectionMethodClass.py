import os
import argparse

from torch.backends import cudnn
from Transformer.utils.utils import *

from Transformer import TransformerSolver
from DeepAnt import DeepAntSolver
from LSTMEncoderDecoder import LSTMSolver
from USAD import USADSolver

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from models import DeepAnt, UsadModel
from datafactory import MyDataset
from sklearn.metrics import classification_report
import json
import plotly.express as px
import plotly.graph_objects as go
import time
from torchinfo import summary


class ADMethod_back():
	def __init__(self, method_name:str, settings: dict, data_path: str):
		self.method_name = method_name
		self.settings = settings
		self.data_path = data_path

	def prepare_pipeline(self):
		if self.method_name == 'transformer':
			self.solver = TransformerSolver.TransformerSolver(self.settings)
		if self.method_name == 'DeepAnt':
			self.solver =  DeepAntSolver.DeepAntSolver(self.settings, self.data_path)
		if self.method_name == 'USAD':
			self.solver =  USADSolver.USADSolver(self.settings)


	def train(self):
		if self.method_name == 'transformer':
			self.solver.train()
		if self.method_name == 'DeepAnt':
			l = self.solver.train()
			return l
		if self.method_name == 'USAD':
			h = self.solver.train()
			return h

	def test(self):
		if self.method_name == 'transformer':
			self.anomaly_scores, self.classification_report = self.solver.test()

		if self.method_name == 'DeepAnt':
			gen_sig, self.anomaly_scores, self.classification_report = self.solver.test()

		if self.method_name == 'USAD':
			self.anomaly_scores, self.classification_report = self.solver.test(alpha = self.settings['ALPHA'], beta = self.settings['BETA'])
		
		
		return self.anomaly_scores, self.classification_report


class ADMethod():
	def __init__(self, name:str, config: dict):
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.name = name
		self.config = config
		if self.config['VERBOSE']:
			print("=====================================================================")
			print("Initializing...")
		start_time = time.time()
		self.train_ds = MyDataset(dataset=self.config['DATASET'], 
								  data_path=self.config['DATAPATH'], 
								  seq_len= self.config['SEQ_LEN'], 
								  step=self.config['STEP'], 
								  method= self.name, 
								  mode='TRAIN',
								 hidden_size = self.config['HIDDEN_SIZE'])
		self.test_ds = MyDataset(dataset=self.config['DATASET'], 
								 data_path=self.config['DATAPATH'], 
								 seq_len= self.config['SEQ_LEN'], 
								 step=self.config['STEP'], 
								 method= self.name,
								 mode='TEST',
								 hidden_size = self.config['HIDDEN_SIZE'])
		self.train_dl = DataLoader(self.train_ds, batch_size = 256, num_workers = 4, shuffle = False)
		self.test_dl = DataLoader(self.test_ds, batch_size = 1, num_workers = 4, shuffle = False)
		
		if self.name == 'DEEPANT':
			self.model = DeepAnt(n_features = self.train_ds.n_features, seq_len = self.config['SEQ_LEN'])
			self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config['LR'], momentum=0.9)
			self.criterion = torch.nn.L1Loss()

		if self.name == 'USAD':
			self.w_size, self.z_size = self.train_ds.get_sizes()
			self.model = UsadModel(w_size = self.w_size, z_size = self.z_size)
			self.optimizer1 = torch.optim.Adam(list(self.model.encoder.parameters())+list(self.model.decoder1.parameters()))
			self.optimizer2 = torch.optim.Adam(list(self.model.encoder.parameters())+list(self.model.decoder2.parameters()))
			

		end_time = time.time()
		elapsed = end_time - start_time
		if self.config['VERBOSE']:
			print(f"Data preprocessing and method configuration finished in {elapsed} sec.")
			print("Model summary: ")
			print(summary(self.model))
			
	def train(self):
		train_history={}
		if[self.config['VERBOSE']]:
			print("=====================================================================")
			print("Training...")
		start_time = time.time()
		### Calling training function based on the model selected+
		if self.name == "DEEPANT":
			train_losses = []
			self.model.to(self.device)
			self.model.train()

			for epoch in range(self.config['EPOCHS']):
				epoch_loss = deepAntEpoch(self.model, self.train_dl, self.criterion, self.optimizer, self.device)
				if self.config['VERBOSE']:
					print(f"Epoch {epoch+1}/{self.config['EPOCHS']}: train_loss:{epoch_loss}")
				train_losses.append([epoch_loss])
			train_history['TRAIN_LOSSES'] = np.array(train_losses)
		if self.name == "USAD":
			train_losses1 = []
			train_losses2 = []
			self.model.to(self.device)
			self.model.train()

			for epoch in range(self.config['EPOCHS']):
				epoch_loss1, epoch_loss2 = UsadEpoch(self.model, self.train_dl, self.optimizer1, self.optimizer2, epoch, self.device)
				if self.config['VERBOSE']:
					print(f"Epoch {epoch+1}/{self.config['EPOCHS']}: train_loss_1:{epoch_loss1}. train_loss_2:{epoch_loss2}")
				train_losses1.append([epoch_loss1])
				train_losses2.append([epoch_loss2])
			train_history['TRAIN_LOSSES_1'] = np.array(train_losses1)
			train_history['TRAIN_LOSSES_2'] = np.array(train_losses2)

		end_time = time.time()
		total_elapsed = end_time - start_time
		if self.config['VERBOSE']:
			print(f"Training finished in {total_elapsed} sec., avg time per epoch: {total_elapsed/self.config['EPOCHS']} sec.")
			print("=====================================================================")
		
		return train_history

	def test(self):
		if[self.config['VERBOSE']]:
			print("=====================================================================")
			print("Testing...")
		start_time = time.time()
		with torch.no_grad():
			### Calling testing function based on the model selected
			if self.name == "DEEPANT":
				self.model.eval()
				self.model.to(self.device)
				self.predictions, self.scores = testDeepAnt(self.model, self.test_dl, self.criterion, self.device)
			
			if self.name == "USAD":
				self.predictions = None #USAD does not return predictions, only returns anomaly scores
				self.model.eval()
				self.model.to(self.device)
				self.scores = testUsad(self.model, self.test_dl, self.device)



		end_time = time.time()
		total_elapsed = end_time - start_time
		if[self.config['VERBOSE']]:
			print(f"Test finished in {total_elapsed} sec.")
			print("=====================================================================")
		return self.predictions, self.scores


	def results(self, threshold: float, plot: bool):
		if self.config['VERBOSE']:
			print("=====================================================================")
			print("Computing results... ") 
		if self.name == "DEEPANT":
			window_size = self.config['SEQ_LEN']
			step = self.config['STEP']
			if self.config['DATASET'] == "SWAT":
				attack = pd.read_csv("./data/SWAT/SWaT_Dataset_Attack_v0.csv", sep=";")
				ground_truth = attack.values[:,-1]
				ground_truth = np.array([False if el == "Normal" else True for el in ground_truth])
				ground_windows = ground_truth[np.arange(window_size)[None, :] + np.arange(0,ground_truth.shape[0]-window_size, step)[:, None]]
				self.ground = np.array([True if el.sum() > 0 else False for el in ground_windows])
			elif self.config['DATASET'] == "NAB":
				with open("./NAB/combined_windows.json") as FI:
					j_label = json.load(FI)
				key = self.config['DATAPATH']
				windows = j_label[key]
				df = pd.read_csv("./NAB/" + self.config['DATAPATH'])
				df['timestamp'] = pd.to_datetime(df['timestamp'])
				ground_truth = np.zeros(len(df), dtype=bool)
				for w in windows:
					start = pd.to_datetime(w[0])
					end = pd.to_datetime(w[1])
					for idx in df.index:
						if df.loc[idx, 'timestamp'] >= start and df.loc[idx, 'timestamp'] <= end:
							ground_truth[idx] = True
				ground_windows = ground_truth[np.arange(window_size)[None, :] + np.arange(0,ground_truth.shape[0]-window_size, step)[:, None]]
				self.ground = np.array([True if el.sum() > 0 else False for el in ground_windows])
			elif self.config['DATASET'] == "MSL":
				ground_truth = np.load("./data/MSL/MSL_test_label.npy")
				ground_windows = ground_truth[np.arange(window_size)[None, :] + np.arange(0,ground_truth.shape[0]-window_size, step)[:, None]]
				self.ground = np.array([True if el.sum() > 0 else False for el in ground_windows])
			elif self.config['DATASET'] == "SMD":
				ground_truth = np.load("./data/SMD/SMD_test_label.npy")
				ground_windows = ground_truth[np.arange(window_size)[None, :] + np.arange(0,ground_truth.shape[0]-window_size, step)[:, None]]
				self.ground = np.array([True if el.sum() > 0 else False for el in ground_windows])

		if self.name == "USAD":
			window_size = self.config['SEQ_LEN']
			step = self.config['STEP']
			if self.config['DATASET'] == "SWAT":
				attack = pd.read_csv("./data/SWAT/SWaT_Dataset_Attack_v0.csv", sep=";")
				ground_truth = attack.values[:,-1]
				ground_truth = np.array([False if el == "Normal" else True for el in ground_truth])
				limit = 0.7*len(ground_truth)
				ground_windows = ground_truth[np.arange(window_size)[None, :] + np.arange(0,limit-window_size, step)[:, None]]
				self.ground = np.array([True if el.sum() > 0 else False for el in ground_windows])
			elif self.config['DATASET'] == "NAB":
				with open("./NAB/combined_windows.json") as FI:
					j_label = json.load(FI)
				key = self.config['DATAPATH']
				windows = j_label[key]
				df = pd.read_csv("./NAB/" + self.config['DATAPATH'])
				df['timestamp'] = pd.to_datetime(df['timestamp'])
				ground_truth = np.zeros(len(df), dtype=bool)
				for w in windows:
					start = pd.to_datetime(w[0])
					end = pd.to_datetime(w[1])
					for idx in df.index:
						if df.loc[idx, 'timestamp'] >= start and df.loc[idx, 'timestamp'] <= end:
							ground_truth[idx] = True
				ground_windows = ground_truth[np.arange(window_size)[None, :] + np.arange(0,ground_truth.shape[0]-window_size, step)[:, None]]
				self.ground = np.array([True if el.sum() > 0 else False for el in ground_windows])
			elif self.config['DATASET'] == "MSL":
				ground_truth = np.load("./data/MSL/MSL_test_label.npy")
				ground_windows = ground_truth[np.arange(window_size)[None, :] + np.arange(0,ground_truth.shape[0]-window_size, step)[:, None]]
				self.ground = np.array([True if el.sum() > 0 else False for el in ground_windows])
			elif self.config['DATASET'] == "SMD":
				ground_truth = np.load("./data/SMD/SMD_test_label.npy")
				ground_windows = ground_truth[np.arange(window_size)[None, :] + np.arange(0,ground_truth.shape[0]-window_size, step)[:, None]]
				self.ground = np.array([True if el.sum() > 0 else False for el in ground_windows])



		thresh = np.percentile(combined_energy, 100 - self.config['CENTILE'])
		self.anomalies = np.array([True if el > threshold else False for el in s])
		self.report = classification_report(self.ground, self.anomalies, output_dict=True)

		# scaler = MinMaxScaler()
		# s = scaler.fit_transform(np.array(self.scores).reshape(-1, 1))
		# self.anomalies = np.array([True if el > threshold else False for el in s])
		# self.report = classification_report(self.ground, self.anomalies, output_dict=True)
		if self.config['VERBOSE']:
			print(classification_report(self.ground, self.anomalies, output_dict=False))
		if plot:
			s_scaled = s.reshape(-1)
			tp = np.logical_and(np.array(s_scaled) >= threshold,  self.ground == True)
			fn = np.logical_and(np.array(s_scaled) < threshold,  self.ground == True)
			fp = np.logical_and(np.array(s_scaled) >= threshold,  self.ground == False)


			fig = go.Figure()
			fig.add_hline(y=threshold, line_color="#aaaaaa", name="threshold", line_dash="dash")
			fig.add_trace(go.Scatter(x=np.arange(len(s)), y=s_scaled, mode='lines', name='Anomaly score', line_color="#515ad6"))
			fig.add_trace(go.Scatter(x=np.arange(len(s))[tp], y=self.ground[tp]*s_scaled[tp], mode='markers', name='True positives', line_color="green"))
			fig.add_trace(go.Scatter(x=np.arange(len(s))[fn], y=self.ground[fn]*threshold, mode='markers', name='False negatives', line_color="red"))
			fig.add_trace(go.Scatter(x=np.arange(len(s))[fp], y=s_scaled[fp], mode='markers', name='False positives', line_color="#000000", marker=dict(size=10, symbol=34, line=dict(width=2))))
			fig.update_layout(plot_bgcolor="white")
			fig.update_xaxes(showgrid=False, gridwidth=1, gridcolor='LightGrey', showline=True, linewidth=2, linecolor='DarkGrey')
			fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey', showline=True, linewidth=2, linecolor='DarkGrey')
			fig.show()

		return self.report




def deepAntEpoch(model: DeepAnt, loader: DataLoader, criterion, optimizer, device):
	curr_loss = 0
	for idx, (batch, batch_labels) in enumerate(loader):
		batch = batch.to(device).permute(0,-1,1)
		batch_labels = batch_labels.to(device)
		output = model(batch)
		loss = criterion(output, batch_labels)
		curr_loss += loss.item()
		loss.backward()
		optimizer.step()
	return curr_loss/len(loader)

def testDeepAnt(model: DeepAnt, loader: DataLoader, criterion, device):
	with torch.no_grad():
		scores = []
		predictions = []
		for i, (batch, batch_labels) in enumerate(loader):
			batch = batch.to(device).permute(0,-1,1)
			batch_labels = batch_labels.to(device)
			output = model(batch)
			score = torch.linalg.norm(output - batch_labels)
			predictions.append(output.cpu().numpy())
			scores.append(score.cpu().item())
	return predictions, scores


def UsadEpoch(model: UsadModel, loader: DataLoader, optimizer1, optimizer2, epoch, device):
	n = epoch+1
	curr_loss1 = 0
	curr_loss2 = 0
	for idx, batch in enumerate(loader):
		batch = batch.to(device)

		#Train AE1
		z = model.encoder(batch)
		w1 = model.decoder1(z)
		w2 = model.decoder2(z)
		w3 = model.decoder2(model.encoder(w1))
		loss1 = 1/n*torch.mean((batch-w1)**2)+(1-1/n)*torch.mean((batch-w3)**2)
		loss2 = 1/n*torch.mean((batch-w2)**2)-(1-1/n)*torch.mean((batch-w3)**2)
		curr_loss1 += loss1.item()
		loss1.backward()
		optimizer1.step()
		optimizer1.zero_grad()

		#Train AE2
		z = model.encoder(batch)
		w1 = model.decoder1(z)
		w2 = model.decoder2(z)
		w3 = model.decoder2(model.encoder(w1))
		loss1 = 1/n*torch.mean((batch-w1)**2)+(1-1/n)*torch.mean((batch-w3)**2)
		loss2 = 1/n*torch.mean((batch-w2)**2)-(1-1/n)*torch.mean((batch-w3)**2)	
		curr_loss2 += loss2.item()
		loss2.backward()
		optimizer2.step()
		optimizer2.zero_grad()

	return curr_loss1/len(loader), curr_loss2/len(loader)

def testUsad(model: UsadModel, loader: DataLoader, device, alpha = 0.5, beta=0.5):
	with torch.no_grad():
		r = []
		for [batch] in loader:
			batch= batch.to(device)
			w1=model.decoder1(model.encoder(batch)).to('cpu')
			w2=model.decoder2(model.encoder(w1.to(device))).to('cpu')
			r.append((alpha*torch.mean((batch.to('cpu')-w1)**2,axis=0)+beta*torch.mean((batch.to('cpu')-w2)**2,axis=0)).to('cpu'))
		
		scores = np.concatenate([torch.stack(r[:-1]).flatten().detach().cpu().numpy(),
							r[-1].flatten().detach().cpu().numpy()])
		
	return scores