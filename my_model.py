import torch.nn as nn
import torch
# from transformer import Block
from torch.nn.modules.rnn import LSTM
from net_modules import *
import numpy as np
from torchinfo import summary

device = "cuda:0"

Frontal = [0,1,21,20,2,3,22,4,24,5,23,25,6,26]
LeftTemporal = [7,8,9,10,11,12,13,14,15,16,17,18,19]
RightTemporal = [27,29,28,30,32,31,33,35,34,36,38,37,39]
channel_seq = (np.array([1,2,3,4,5,6,7,8,9,10,11,12,14,15,13,17,18,21,23,24,27,29,30,33,35,36,39,16,20,19,22,26,25,28,32,31,34,38,37,40])-1).tolist()
# print(channel_seq[28])
channel_left = (np.array([1,2,5,8,10,14,13,17,18,21,23,24,27,29,30,33,35,36,39]) - 1).tolist()
channel_right = (np.array([4,3,7,9,12,15,16,20,19,22,26,25,28,32,31,34,38,37,40]) - 1).tolist()

class Conv1dWithConstraint(nn.Conv1d):
    def __init__(self, *args, max_norm = 1, **kwargs):
        self.max_norm = max_norm
        super(Conv1dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super(Conv1dWithConstraint, self).forward(x)

class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=4):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)				# 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        weight = y
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class fnirsNet_time(nn.Module):
    def __init__(self, n_class, channels, samples):
        super().__init__()

        self.channels = channels
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 8), stride=(1, 4)),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=(1, 4), stride=(1, 2)),
            nn.ELU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(64*8*8, n_class),
        )

    def forward(self, x):
        x = x[:, channel_seq, :]
        input = x.unsqueeze(1)

        feature0 = input
        feature = self.block1(feature0)
        logits = self.classifier(feature)
        return logits

class fnirsNet_spatial(nn.Module):
    def __init__(self, n_class, channels, samples):
        super().__init__()

        self.channels = channels
        self.block2 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(8, 1), stride=(4, 1)),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=(4, 1), stride=(2, 1)),
            nn.ELU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(64*8*8, n_class),
        )

    def forward(self, x):
        x = x[:, channel_seq, :]
        input = x.unsqueeze(1)

        feature0 = input
        feature = self.block2(feature0)
        logits = self.classifier(feature)
        return logits

class mean(nn.Module):
    def __init__(self, n_class, channels, samples):
        super().__init__()

        self.channels = channels
        self.fc1 = nn.Sequential(
            nn.Linear(channels, 64),
            nn.ELU(inplace=True),
            nn.Linear(64, 128),
            nn.ELU(inplace=True),
            nn.Linear(128, 256)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, n_class),
        )

    def forward(self, x):
        x = x[:, channel_seq, :]
        mean = torch.mean(x, dim=-1)
        feature = self.fc1(mean)

        logits = self.classifier(feature)
        return logits

class DBJNet(nn.Module):
    def __init__(self, n_class, channels, samples):
        super().__init__()

        self.channels = channels
        self.block2 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(8, 1), stride=(4, 1)),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=(4, 1), stride=(2, 1)),
            nn.ELU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(channels, 64),
            nn.ELU(inplace=True),
            nn.Linear(64, 128),
            nn.ELU(inplace=True),
            nn.Linear(128, 256)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(64*8*8, 256),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, n_class),
        )

    def forward(self, x):
        x = x[:, channel_seq, :]
        mean = torch.mean(x, dim=-1)
        input = x.unsqueeze(1)

        feature0 = input
        feature = self.block2(feature0)
        feature1 = torch.nn.functional.normalize(self.fc1(mean), p=2, dim=1)
        feature2 = torch.nn.functional.normalize(self.fc2(feature), p=2, dim=1)
        feature = torch.cat((feature1, feature2), dim=1)
        logits = self.classifier(feature)
        return logits


class CNN_NLSTM(nn.Module):
    def __init__(self, n_class, channels, samples):
        super().__init__()

        self.channels = channels
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(8, 1), stride=(4, 1)),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=(4, 1), stride=(2, 1)),
            nn.ELU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(1, 8), stride=(1, 4)),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(1, 4), stride=(1, 2)),
            nn.ELU(inplace=True),
        )
        self.lstm = LSTM(64, 64, 1, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*54, n_class),
        )

    def forward(self, x):
        x = x[:, channel_seq, :]
        input = x.unsqueeze(1)
        feature0 = input
        feature = self.block1(feature0)
        feature = self.block2(feature)
        feature = feature.reshape(feature.shape[0], 64, -1).permute(0, 2, 1)
        feature, _ = self.lstm(feature)

        logits = self.classifier(feature)
        return logits

if __name__ == '__main__':
    with torch.no_grad():
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        cuda0 = torch.device('cuda:0')
        x = torch.rand((16, 40, 160), device=cuda0)
        model = fnirsNet(n_class=3, channels=40, samples=160)
        summary(model, input_size=(16, 40, 160))
        model.cuda()
        output = model(x)
        print('output:', output.shape)



