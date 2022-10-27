import os
import argparse

from torch.backends import cudnn
from Transformer.utils.utils import *

from Transformer import TransformerSolver
from DeepAnt import DeepAntSolver


class ADMethod():
	def __init__(self, method_name:str, settings: dict, data_path: str):
		self.method_name = method_name
		self.settings = settings
		self.data_path = data_path

	def prepare_pipeline(self):
		if self.method_name == 'transformer':
			self.solver = TransformerSolver.Solver(self.settings)
		if self.method_name == 'DeepAnt':
			self.solver =  DeepAntSolver.Solver(self.settings, self.data_path)


	def train(self):
		if self.method_name == 'transformer':
			self.solver.train()
		if self.method_name == 'DeepAnt':
			l = self.solver.train()
			return l

	def test(self):
		if self.method_name == 'transformer':
			self.solver.test()
		if self.method_name == 'DeepAnt':
			p, l = self.solver.test()
			return p, l

	def results(self):
		pass