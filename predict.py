import librosa
import numpy as np
import glob
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import random
from tqdm import tqdm 
import json

from utils import *
from model import *

import argparse

parser = argparse.ArgumentParser(description = 'Predicts values')
parser.add_argument('-tst', '--test_data', type = str, metavar = '', required = True, help = 'path to test_evaluationlib dataset (location of test_evaluationlib directory)')
parser.add_argument('-d', '--device', type = str, metavar = '', required = True, help = 'device to use for model')

args = parser.parse_args()

def predict_values(args):

	device = args.device

	model = torch.load('best_model.pth')

	transforms = Compose([MFCC_transform()])
	test_dataset = EvaluationDataset(args.test_data, transforms = transforms)
	test_dataloader = DataLoader(test_dataset, collate_fn = collate_fn_test, batch_size = 64, shuffle = False, drop_last = False)
	
	output = []
	with torch.no_grad():
		for batch in tqdm(test_dataloader):
			X, mask = batch
			X = X.to(device)

			proba = model(X).sigmoid()
			proba = proba.cpu().numpy().tolist()

			for i in range(len(proba)):
				output.append(proba[i][:mask[i]])


	final_output = []
	for el in output:
		final_output.append([frame[0] for frame in el])

	output = dict()
	for i in range(len(test_dataset.annot)):
		output[test_dataset.annot[i].split('/')[-1].split('\\')[-1]] = final_output[i]

	with open("predictions.json", "w") as write_file:
		json.dump(output, write_file)

if __name__ == '__main__':
	predict_values(args)




