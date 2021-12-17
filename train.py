import librosa
import numpy as np
import glob
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm 

from utils import *
from model import *

import argparse

parser = argparse.ArgumentParser(description = 'Applies training')
parser.add_argument('-lbs', '--librispeech', type = str, metavar = '', required = True, help = 'path to LibriSpeech dataset (location of LibriSpeech directory)')
parser.add_argument('-m', '--musan', type = str, metavar = '', required = True, help = 'path to musan dataset (location of musan directory)')
parser.add_argument('-n', '--n_epoch', type = int, metavar = '', required = True, help = 'number of epochs to train model')
parser.add_argument('-d', '--device', type = str, metavar = '', required = True, help = 'device to train model')

args = parser.parse_args()

def train(args):

	torch.manual_seed(42)
	np.random.seed(42)
	random.seed(42)

	#setting model and optimizer
	model = Net().to(args.device)
	optimizer = torch.optim.RAdam(model.parameters())

	#splitting noize dataset into train and val
	noise_annot = glob.glob(args.musan + '/musan/noise/free-sound/noise*')+ glob.glob(args.musan + '/musan/noise/sound-bible/noise*')
	train_noise_annot, val_noise_annot = train_test_split(noise_annot, train_size = 0.8)

	#setting transforms
	train_transforms = Compose([Noise(NoiseDataset(train_noise_annot)), MFCC_transform()])
	val_transforms = Compose([Noise(NoiseDataset(val_noise_annot), random = False), MFCC_transform()])

	#setting dataset
	train_dataset = VAD_Dataset(transforms = train_transforms, root = args.librispeech, url = 'train-clean-360')
	val_dataset = VAD_Dataset(transforms = val_transforms, root = args.librispeech, url = 'train-clean-100')

	#setting dataloaders
	train_dataloader = DataLoader(train_dataset, collate_fn = collate_fn_train, batch_size = 64, shuffle  = True, drop_last = True)
	val_dataloader = DataLoader(val_dataset, collate_fn = collate_fn_train, batch_size = 64, shuffle = False, drop_last = False)

	prev_score = 0.
	for i in tqdm(range(args.n_epoch)):
		loss, train_score = training_epoch(model, optimizer, train_dataloader, args.device)
		val_score = validate_epoch(model, val_dataloader, args.device)
		if val_score > prev_score:
			torch.save(model, 'best_model.pth')
		print('EPOCH {} | Train log_loss {} | Train score {} | Val score {}'.format(i+1, loss, train_score, val_score))

if __name__ == '__main__':
	train(args)