import os
import argparse

from torch.backends import cudnn
from Transformer.utils.utils import *

from Transformer import TransformerSolver
from DeepAnt import DeepAntSolver
from LSTMEncoderDecoder import LSTMSolver
from USAD import USADSolver


class ADMethod():
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
			self.solver.test()
		if self.method_name == 'DeepAnt':
			p, l = self.solver.test()
			return p, l
		if self.method_name == 'USAD':
			r = self.solver.test()
			return r


	def results(self):
		pass