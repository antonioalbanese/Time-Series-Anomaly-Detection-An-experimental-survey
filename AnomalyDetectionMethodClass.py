import os
import argparse

from torch.backends import cudnn
from Transformer.utils.utils import *

from Transformer import TransformerSolver
from DeepAnt import DeepAntSolver
from LSTMEncoderDecoder import LSTMSolver
from USAD import USADSolver



from torch.utils.data import DataLoader
from models import DeepAnt
from datafactory import MyDataset


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
		self.train_ds = MyDataset(dataset=self.config['DATASET'], 
								  data_path=self.config['DATAPATH'], 
								  seq_len= self.config['SEQ_LEN'], 
								  step=self.config['STEP'], 
								  method= self.name, 
								  mode='TRAIN')
		self.test_ds = MyDataset(dataset=self.config['DATASET'], 
								 data_path=self.config['DATAPATH'], 
								 seq_len= self.config['SEQ_LEN'], 
								 step=self.config['STEP'], 
								 method= self.name,
								 mode='TEST')
		self.train_dl = DataLoader(self.train_ds, batch_size = 256, num_workers = 4, shuffle = False)
		self.test_dl = DataLoader(self.test_ds, batch_size = 1, num_workers = 4, shuffle = False)
		
		if self.name == 'DEEPANT':
			self.model = DeepAnt(n_features = self.train_ds.n_features, seq_len = self.config['SEQ_LEN'])
			self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config['LR'], momentum=0.9)
			self.criterion = torch.nn.L1Loss()
			
	def train(self):
		self.model.to(self.device)
		self.model.train()

		for epoch in range(self.config['EPOCHS']):
			epoch_loss = deepAntEpoch(self.model, self.train_dl, self.criterion, self.optimizer, self.device)
			if self.config['VERBOSE']:
				print(f"Epoch {epoch+1}/{self.config['EPOCHS']}: train_loss:{epoch_loss}")
		if self.config['VERBOSE']:
			print("training finished")

	def test(self):
		with torch.no_grad():
			self.model.eval()
			self,model.to(device)
			predictions, scores = testDeepAnt(self.model, self.test_dl, self.criterion, self.device)
		return predictions, scores








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
