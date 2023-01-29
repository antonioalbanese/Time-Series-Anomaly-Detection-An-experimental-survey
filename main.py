import os
import sys
import argparse
import random

import zipfile
import numpy as np
import subprocess

from AnomalyDetectionMethodClass import ADMethod

method_list = ["DEEPANT", "TANOGAN", "USAD", "TRANSFORMER"]

subprocess.run(["mkdir", "./data"])

parser = argparse.ArgumentParser()
parser.add_argument("METHOD", required = True)
parser.add_argument("DATASET", required = True)
parser.add_argument("NAB_PATH", default="None", required = True)
parser.add_argument("SEQ_LEN", type=int, required = True)
parser.add_argument("STEP", type=int, required = True)
parse.add_argument("LR", type=float, required = True)
parser.add_argument("EPOCHS", type=int, required = True)
parser.add_argument("VERBOSE", type=bool, default=False, required = True)
parser.add_argument("LOGGER", type=bool, default = True, required = True)
parser.add_argument("HIDDEN_SIZE", type=int, default=None, required = True)
parser.add_argument("SEED", type=int, required = True)
parser.add_argument("DOWNLOAD_DS", type=bool, default=False, required = True)

args = parser.parse_args()

if args.DOWNLOAD_DS:
    if args.DATSET == "SWAT":
        subprocess.run(["mkdir", "./data/SWAT"])
        #normal
        subprocess.run(["python", "USAD/gdrivedl.py https://drive.google.com/open?id=1rVJ5ry5GG-ZZi5yI4x9lICB8VhErXwCw data/SWAT"])
        #anomalies
        subprocess.run(["python", "USAD/gdrivedl.py https://drive.google.com/open?id=1iDYc0OEmidN712fquOBRFjln90SbpaE7 data/SWAT"])
    if args.DATASET == "SMD":
        subprocess.run(["mkdir", "./data/SMD"])
        subprocess.run(["python", "USAD/gdrivedl.py https://drive.google.com/file/d/18JNYBsaX7tu0Qfgo92nCBCklv471L1xB data/SMD"])
        with zipfile.ZipFile("./data/SMD/SMD.zip", 'r') as zip_ref:
            zip_ref.extractall("./data/")
    if args.DATASET == "MSL":
        subprocess.run(["mkdir", "./data/MSL"])
        subprocess.run(["python", "USAD/gdrivedl.py https://drive.google.com/uc?id=1ZCLBU_pKTbsPlcj_LwxZE3IRy6mrlys3 data/MSL"])
        with zipfile.ZipFile("./data/MSL/MSL.zip", 'r') as zip_ref:
            zip_ref.extractall("./data/")
    if args.DATASET == "NAB":
        pass 

random.seed(args.SEED)
configuration = {
    'DATASET': args.DATASET,
    'DATAPATH': args.NAB_PATH,
    'SEQ_LEN': args.SEQ_LEN,
    'STEP': args.STEP,
    'HIDDEN_SIZE': args.HIDDEN_SIZE, 
    'LR': args.LR,
    'EPOCHS': args.EPOCHS,
    'VERBOSE': args.VERBOSE,
    'LOGGER': args.LOGGER
}

method = ADMethod(name = args.METHOD, config = configuration)
train_losses = method.train()
predictions, score = method.test()
for th in np.linspace(0,1,10,endpoint=False):
  rep = method.results(threshold = th, plot = False)
method.close_run()