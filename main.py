import os
import sys
import argparse
import random

import zipfile
import numpy as np
import subprocess

from AnomalyDetectionMethodClass import ADMethod

method_list = ["DEEPANT", "TANOGAN", "USAD", "TRANSFORMER"]



parser = argparse.ArgumentParser()
parser.add_argument("--5-METHOD")
parser.add_argument("--3-DATASET")
parser.add_argument("--4-NAB_PATH", default="None")
parser.add_argument("--2-SEQ_LEN", type=int)
parser.add_argument("--1-SEED", type=int)

args = parser.parse_args()

# if args.DOWNLOAD_DS:
#    subprocess.run(["mkdir", "./data"])
#     if args.DATASET == "SWAT":
#         subprocess.run(["mkdir", "./data/SWAT"])
#         #normal
#         subprocess.run(["python", "USAD/gdrivedl.py https://drive.google.com/open?id=1rVJ5ry5GG-ZZi5yI4x9lICB8VhErXwCw data/SWAT"])
#         #anomalies
#         subprocess.run(["python", "USAD/gdrivedl.py https://drive.google.com/open?id=1iDYc0OEmidN712fquOBRFjln90SbpaE7 data/SWAT"])
#     if args.DATASET == "SMD":
#         subprocess.run(["mkdir", "./data/SMD"])
#         subprocess.run(["python", "USAD/gdrivedl.py https://drive.google.com/file/d/18JNYBsaX7tu0Qfgo92nCBCklv471L1xB data/SMD"])
#         with zipfile.ZipFile("./data/SMD/SMD.zip", 'r') as zip_ref:
#             zip_ref.extractall("./data/")
#     if args.DATASET == "MSL":
#         subprocess.run(["mkdir", "./data/MSL"])
#         subprocess.run(["python", "USAD/gdrivedl.py https://drive.google.com/uc?id=1ZCLBU_pKTbsPlcj_LwxZE3IRy6mrlys3 data/MSL"])
#         with zipfile.ZipFile("./data/MSL/MSL.zip", 'r') as zip_ref:
#             zip_ref.extractall("./data/")
#     if args.DATASET == "NAB":
#         pass 

random.seed(args.SEED)

match args.METHOD:
  case "TRANSFORMER":
    configuration = {
      'DATASET': args.DATASET,
      'DATAPATH': args.NAB_PATH,
      'SEQ_LEN': args.SEQ_LEN,
      'STEP': args.SEQ_LEN,
      'LR': 0.001,
      'EPOCHS': 20,
      'VERBOSE': False,
      'LOGGER': True
    }
  case "TANO":
    configuration = {
      'DATASET': args.DATASET,
      'DATAPATH': args.NAB_PATH,
      'SEQ_LEN': args.SEQ_LEN,
      'STEP': args.SEQ_LEN,
      'LR': 0.0001,
      'EPOCHS': 30,
      'VERBOSE': False,
      'LOGGER': True,
      'K': 3 
    }
  case "DEEPANT":
    configuration = {
      'DATASET': args.DATASET,
      'DATAPATH': args.NAB_PATH,
      'SEQ_LEN': args.SEQ_LEN,
      'STEP': args.SEQ_LEN,
      'LR': 0.000001,
      'EPOCHS': 130,
      'VERBOSE': False,
      'LOGGER': True
    }
  case "USAD":
    configuration = {
      'DATASET': args.DATASET,
      'DATAPATH': args.NAB_PATH,
      'SEQ_LEN': args.SEQ_LEN,
      'STEP': args.SEQ_LEN,
      'HIDDEN_SIZE': 100, 
      'LR': 0.0001,
      'EPOCHS': 30, 
      'VERBOSE': False,
      'LOGGER': True
    }
  case _:
    sys.exit("METHOD must be in [TANO, TRANSOFRMER, DEEPANT, USAD]")


method = ADMethod(name = args.METHOD, config = configuration)
train_losses = method.train()
predictions, score = method.test()


for th in np.linspace(0,1,10,endpoint=False):
  rep = method.results(threshold = th, plot = False)
method.close_run()