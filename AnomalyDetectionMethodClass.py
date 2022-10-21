import os
import argparse

from torch.backends import cudnn
from Transformer.utils.utils import *

from Transformer.solver import Solver


class ADMethod():
	def __init__(self, method_name:str, settings: dict, data_path: str):
		self.method_name = method_name
		self.settings = settings
		self.data_path = data_path

	def prepare_pipeline(self):
		if self.method_name == 'transformer':
			self.solver = Solver(vars(config))


	def train(self):
		if self.method_name == 'transformer':
			self.solver.train()


	def test(self):
		if self.method_name == 'transformer':
			self.solver.test()

	def results(self):
		pass