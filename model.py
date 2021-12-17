import torch
import torch.nn as nn


class Frequential_attention(nn.Module):
    def __init__(self, input_dim=13, emb_size=16):
        super().__init__()
        self.conv1 = nn.Conv2d(1, emb_size, kernel_size=(input_dim, 1))
        self.conv2 = nn.Conv2d(1, emb_size, kernel_size=(emb_size, 1))

    def forward(self, x):
        x = self.conv1(x.unsqueeze(1)).squeeze(2).relu()
        att = self.conv2(x.unsqueeze(1)).squeeze(2)
        att = nn.functional.softmax(att, dim=1)
        x = (x * att)
        return x

class Net(nn.Module):
    def __init__(self, emb_size = 16):
        super().__init__()
        
        self.freq_att_1 = Frequential_attention(13, emb_size)
        self.conv2d = nn.Conv2d(1, emb_size//2, kernel_size = (2, 2))
        self.pool = nn.AdaptiveMaxPool1d(8)
        self.freq_att_2 = Frequential_attention(emb_size // 2, emb_size // 4)
        self.lstm = nn.LSTM(emb_size // 4, emb_size // 4, batch_first = True)
        self.cls = nn.Linear(emb_size // 4, 1)
        
    def forward(self, x):
        x = self.freq_att_1(x)
        x = self.conv2d(x.unsqueeze(1)).relu()
        x = x.reshape(x.shape[0], -1, x.shape[3]).permute(0, 2, 1)

        x = self.pool(x).permute(0, 2, 1)
        x = self.freq_att_2(x).relu()
        x, _ = self.lstm(x.permute(0,2,1))
        x = self.cls(x)
        return x