import os
import glob

import numpy as np
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
from torchaudio.datasets import LIBRISPEECH
import librosa
from tqdm import tqdm

class NoiseDataset:
    def __init__(self, annot):
        self.annot = annot
        
    def __len__(self):
        return len(self.annot)
    
    def getitem(self, index, sr = None):
        return librosa.load(self.annot[index], sr = sr)

class EvaluationDataset:
    """Evaluation dataset"""
    def __init__(self, noise_path = os.getcwd(), transforms = None):
        self.path = noise_path
        self.transforms = transforms
        self.annot = glob.glob(self.path + '/test_evaluationlib/*')

    def __len__(self):
        return len(self.annot)

    def __getitem__(self, index):
        wav, sr = librosa.load(self.annot[index], sr = None)

        hop_length = int(sr * 10 / 1000)
        total_len = int(np.ceil(len(wav) / hop_length))
        n_fft = 2 * hop_length

        if self.transforms is None:
            return wav, sr
        else:
            return self.transforms(wav, sr, hop_length, n_fft, total_len), total_len

class VAD_Dataset(LIBRISPEECH):
    """LIBRISPEECH dataset"""
    def __init__(self, transforms, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transforms = transforms
        
        
    def __getitem__(self, index):
        wav, sr, _, _, _, _ = super().__getitem__(index)
        hop_length = int(sr * 10 / 1000) #10 ms
        n_fft = 2 * hop_length
        wav = wav[0].numpy()
        total_len = int(np.ceil(len(wav) / hop_length))
        tgt = []
        for i in range(0, len(wav), hop_length):
            if np.sum(wav[i : min((i + hop_length), len(wav))] > 0.01) > 0:
                tgt.append(1.)
            else:
                tgt.append(0.)
        if self.transforms is None:
            return wav, sr, np.array(tgt), hop_length, n_fft, total_len
        else:
            return self.transforms(wav, sr, hop_length, n_fft, total_len), np.array(tgt)

def mix_up(wave_1, wave_2, db):
    """mixes 2 wav into one with sound div"""
    wave_1_power = np.linalg.norm(wave_1, ord = 2) 
    wave_2_power = np.linalg.norm(wave_2, ord = 2) 
    snr = np.exp(db / 10)
    scale = (snr * wave_2_power) / wave_1_power
    
    
    if len(wave_1) <= len(wave_2):
        noisy_speech = (wave_1 * scale + wave_2[:len(wave_1)]) / 2
        return noisy_speech 
    
    else:
        noisy_speech = wave_1
        for i in range(0, len(wave_1), len(wave_2)):
            if i + len(wave_2) > len(wave_1):
                noisy_speech[i : len(wave_1)] = (noisy_speech[i : len(wave_1)] * scale + wave_2[: (len(wave_1) - i)]) / 2
            else:
                noisy_speech[i : (i + len(wave_2))] =  (noisy_speech[i : (i + len(wave_2))] * scale + wave_2 ) / 2
        return noisy_speech
    
    
class Compose:
    """Applies transforms"""
    def __init__(self, transforms = None):
        self.transforms = transforms
        
    def __call__(self, wav, *args, **kwargs):
        for i in range(len(self.transforms)):
            wav = self.transforms[i](wav, *args, **kwargs)
        return wav
    
    
class MFCC_transform:
    """Transforms wav to MFCC"""
    def __init__(self, n_mfcc = 13):
        self.n_mfcc = n_mfcc
    
    def __call__(self, wav, *args, **kwargs):
        sr, hop_length, n_fft, total_len = args
        return librosa.feature.mfcc(wav, sr = sr, hop_length = hop_length, n_fft = n_fft, n_mfcc=self.n_mfcc)[:,:total_len]

class Noise:
    """Applies noise to wav randomly"""
    def __init__(self, noise_dataset, db = [20, 10, 0., -10], random = True):
        self.i = 0
        self.db = db
        self.noise_dataset = noise_dataset
        self.random = random
    def __call__(self, wav, sr, *args, **kwargs):
        if self.random:
            
            db = np.random.choice(self.db)
            i = np.random.randint(len(self.noise_dataset))
            noise, _ = self.noise_dataset.getitem(i, sr)
            
            return mix_up(wav, noise, db)
        else:
            
            db = self.db[self.i % len(self.db)]
            noise, _ = self.noise_dataset.getitem(self.i % len(self.noise_dataset) , sr)
            
            self.i = (self.i + 1) % len(self.noise_dataset)
            
            return mix_up(wav, noise, db)
            
def collate_fn_train(batch):
    batch_size = len(batch)
    t_max = np.max([el[0].shape[1] for el in batch]) + 1
    h_max = np.max([el[0].shape[0] for el in batch])
    t_max_tgt = np.max([len(el[1]) for el in batch])
    
    output_features = np.zeros((batch_size, h_max, t_max))
    output_tgt = np.zeros((batch_size, t_max_tgt))
    
    for i in range(batch_size):
        features = batch[i][0]
        h, t = features.shape
        tgt_t = batch[i][1]

        output_tgt[i,:len(tgt_t)] = tgt_t 
        output_features[i,:h,1:(t+1)] = features
    return torch.Tensor(output_features), torch.Tensor(output_tgt)


def collate_fn_test(batch):
    batch_size = len(batch)
    t_max = np.max([el[0].shape[1] for el in batch]) + 1
    h_max = np.max([el[0].shape[0] for el in batch])
    
    mask = [el[1] for el in batch]
    
    output_features = np.zeros((batch_size, h_max, t_max))
    
    for i in range(batch_size):
        features = batch[i][0]
        h, t = features.shape
        tgt_t = batch[i][1]

        output_features[i,:h,1:(t+1)] = features
    return torch.Tensor(output_features), mask

def training_epoch(model, optimizer, dataloader, device, criterion = nn.BCEWithLogitsLoss()):
    total_loss = []
    prob = []
    y_true = []
    
    for batch in tqdm(dataloader):
        model.train()
        optimizer.zero_grad()
        
        X, y = batch
        X, y = X.to(device), y.to(device)

        output = model(X)
        loss = criterion(output.squeeze(2), y)
        
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            total_loss.append(loss.detach().cpu().item())
            y_true = y_true + y.reshape(-1).numpy().tolist()
            prob = prob + output.reshape(-1).detach().cpu().numpy().tolist()
    
    return np.mean(total_loss), roc_auc_score(y_true, prob)


def validate_epoch(model, dataloader, device):
    model.eval()
    total_loss = []
    prob = []
    y_true = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            
            X, y = batch
            X, y = X.to(device), y.to(device)

            output = model(X)
            
            prob = prob + output.reshape(-1).numpy().tolist()
            y_true = y_true + y.reshape(-1).numpy().tolist()
    return roc_auc_score(y_true, prob)