# This is the networks script
import math
from collections import OrderedDict

import torch
import torch.nn as nn
from net_modules import *
import numpy as np

channel_seq = (np.array([1,2,3,4,5,6,7,8,9,10,11,12,14,15,13,17,18,21,23,24,27,29,30,33,35,36,39,16,20,19,22,26,25,28,32,31,34,38,37,40])-1).tolist()

class TSception(nn.Module):
    def conv_block(self, in_chan, out_chan, kernel, step, pool):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=out_chan,
                      kernel_size=kernel, stride=step),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(1, pool), stride=(1, pool)))

    def __init__(self, num_classes, input_size, sampling_rate, num_T, num_S, hidden, dropout_rate):
        # input_size: 1 x EEG channel x datapoint
        super(TSception, self).__init__()
        self.inception_window = [0.5, 0.25, 0.125]
        self.pool = 8
        # by setting the convolutional kernel being (1,lenght) and the strids being 1 we can use conv2d to
        # achieve the 1d convolution operation
        self.Tception1 = self.conv_block(1, num_T, (1, int(self.inception_window[0] * sampling_rate)), 1, self.pool)
        self.Tception2 = self.conv_block(1, num_T, (1, int(self.inception_window[1] * sampling_rate)), 1, self.pool)
        self.Tception3 = self.conv_block(1, num_T, (1, int(self.inception_window[2] * sampling_rate)), 1, self.pool)

        self.Sception1 = self.conv_block(num_T, num_S, (int(input_size[1]), 1), 1, int(self.pool*0.25))
        self.Sception2 = self.conv_block(num_T, num_S, (int(input_size[1] * 0.5), 1), (int(input_size[1] * 0.5), 1),
                                         int(self.pool*0.25))
        self.fusion_layer = self.conv_block(num_S, num_S, (3, 1), 1, 4)
        self.BN_t = nn.BatchNorm2d(num_T)
        self.BN_s = nn.BatchNorm2d(num_S)
        self.BN_fusion = nn.BatchNorm2d(num_S)

        self.fc = nn.Sequential(
            nn.Linear(num_S, hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        x = x[:, channel_seq, :]
        x = x.unsqueeze(1)
        y = self.Tception1(x)
        out = y
        y = self.Tception2(x)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(x)
        out = torch.cat((out, y), dim=-1)
        out = self.BN_t(out)
        z = self.Sception1(out)
        out_ = z
        z = self.Sception2(out)
        out_ = torch.cat((out_, z), dim=2)
        out = self.BN_s(out_)
        out = self.fusion_layer(out)
        out = self.BN_fusion(out)
        out = torch.squeeze(torch.mean(out, dim=-1), dim=-1)
        out = self.fc(out)
        return out


class EEGNet_V3x5_Vit(nn.Module):
    def __init__(self, num_classes, input_size, dropout_rate=0.1, sampling_rate=128):
        super().__init__()
        chans: int = input_size[-2]
        samples: int = input_size[-1]
        F1 = chans * 2
        F2 = F1 * 2
        downSample_1, downSample_2 = 4, 4
        kernLength = 15
        kernel_2 = 15
        kernel_2 = kernel_2 if kernel_2 % 2 == 1 else kernel_2 + 1

        # Make kernel size and odd number
        try:
            assert kernLength % 2 != 0
        except AssertionError:
            raise ValueError("ERROR: kernLength must be odd number")

        self.seq_embed = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernLength), padding=(0, kernLength // 2), groups=1),  # conv1 250Hz
            nn.BatchNorm2d(F1),
            nn.ReLU(),
            nn.AvgPool2d((1, downSample_1)),  # 250Hz // downSample_1
            nn.Dropout(dropout_rate),
            SeparableConv2d(F1, F2, kernel_size=(chans, 1)),  # conv3
            nn.BatchNorm2d(F2),
            nn.ReLU(),
            nn.AvgPool2d((1, downSample_2)),  # 250 Hz // downSample_1 // downSample_2
            nn.Dropout(dropout_rate),
            Squeeze(2),
            Permute(dims=[0, 2, 1])
        )

        # transformer
        seq_len = int(samples // (downSample_1 * downSample_2))
        embed_dim = F2
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len + 1, embed_dim))

        self.transformer = TransformerVit(dim=embed_dim, depth=6, heads=8, dim_head=256, mlp_dim=128, dropout=0.)
        # self.layernorm = nn.LayerNorm(embed_dim, eps=1e-12)
        self.layernorm = nn.Identity()

        # In: (B, F2 *  (samples - chans + 1) / 32)
        self.classifier = nn.Linear(embed_dim, num_classes)

    # @timer_wrap
    def forward(self, x: torch.Tensor):
        # Block 1
        seq_embed = self.seq_embed(x)

        batch_size, seq_len, embed_dim = seq_embed.shape
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, seq_embed), dim=1)
        embeddings += self.pos_embedding
        tr_output = self.transformer(embeddings)
        sequence_output = self.layernorm(tr_output)
        cls_token = sequence_output[:, 0, :]
        y2 = self.classifier(cls_token)

        return y2


class EEG_V1x12_Vit(nn.Module):
    def __init__(self, num_classes, input_size, dropout_rate=0.1, sampling_rate=128):
        super().__init__()
        chans: int = input_size[-2]
        samples: int = input_size[-1]
        kernLength = 31
        F1 = chans * 2
        F2 = F1 * 2
        downSample_1, downSample_2 = 4, 5
        kernel_2 = 15

        # Make kernel size and odd number
        try:
            assert kernLength % 2 != 0
        except AssertionError:
            raise ValueError("ERROR: kernLength must be odd number")
        self.depth = [1, 2]
        self.stage0 = nn.Sequential(
            Squeeze(1),
            nn.Conv1d(chans, F1, kernLength // 2, padding=(kernLength // 4), groups=chans),  # conv1
            nn.Conv1d(F1, F1, kernLength // 2, padding=(kernLength // 4), groups=chans),
            nn.BatchNorm1d(F1),
            # nn.Dropout(dropoutRate)
        )
        self.stage1 = nn.ModuleList([])
        for _ in range(self.depth[0]):
            self.stage1.append(
                nn.Sequential(
                    nn.Conv1d(F1, F2, kernLength // 2, groups=F1),  # conv 2
                    nn.Conv1d(F2, F2, kernLength // 2, groups=F2),
                    nn.BatchNorm1d(F2),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                )
            )
        self.stage2 = nn.ModuleList([])
        for _ in range(self.depth[1]):
            self.stage2.append(
                nn.Sequential(
                    SeparableConv1d(F2, F2, kernel_size=31 // 2, padding=31 // 4),  # conv3
                    SeparableConv1d(F2, F2, kernel_size=31 // 2, padding=31 // 4),
                    nn.BatchNorm1d(F2),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                )
            )
        self.norm_s1 = nn.BatchNorm1d(F1)
        self.norm_s2 = nn.BatchNorm1d(F2)

        self.merge_s1 = nn.Sequential(
            nn.AvgPool1d(downSample_1),
            # nn.Conv1d(F1, F2, kernLength // 2, padding=(kernLength // 4), groups=chans)
        )
        self.merge_s2 = nn.Sequential(
            nn.AvgPool1d(downSample_2),
            Permute([0, 2, 1])
        )


        # transformer
        seq_len = math.ceil((samples - kernLength + 1) // (downSample_1 * downSample_2))
        embed_dim = F2
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len + 1, embed_dim))

        self.transformer = TransformerVit(dim=embed_dim, depth=6, heads=8, dim_head=256, mlp_dim=128, dropout=0)
        # self.layernorm = nn.LayerNorm(embed_dim)
        self.layernorm = nn.Identity()

        # In: (B, F2 *  (samples - chans + 1) / 32)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, num_classes),
        )

    # @timer_wrap
    def forward(self, x: torch.Tensor):
        # Block 1
        seq_embed = self.forward_embed(x)

        batch_size, seq_len, embed_dim = seq_embed.shape
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, seq_embed), dim=1)
        embeddings += self.pos_embedding
        tr_output = self.transformer(embeddings)
        sequence_output = self.layernorm(tr_output)
        cls_token = sequence_output[:, 0, :]
        y2 = self.classifier(cls_token)

        return y2

    def forward_embed(self, x):
        x = self.stage0(x)
        for stage1 in self.stage1:
            x = stage1(x)
        x = self.merge_s1(x)
        for stage2 in self.stage2:
            x = stage2(x)
        x = self.merge_s2(x)
        return x


class EEG_V1x14_Vit(nn.Module):
    def __init__(self, num_classes, input_size, dropout_rate=0.5, sampling_rate=128):
        super().__init__()
        chans: int = input_size[-2]
        samples: int = input_size[-1]
        F1 = chans * 2
        F2 = F1 * 2
        downSample_1, downSample_2 = 4, 5
        kernLength = 31
        kernel_2 = 15

        # Make kernel size and odd number
        try:
            assert kernLength % 2 != 0
        except AssertionError:
            raise ValueError("ERROR: kernLength must be odd number")

        self.depth = [1, 2]
        self.stage0 = nn.Sequential(
            Squeeze(1),
            nn.Conv1d(chans, F1, kernLength // 2, padding=(kernLength // 4), groups=chans),  # conv1
            nn.Conv1d(F1, F1, kernLength // 2, padding=(kernLength // 4), groups=chans),
            nn.BatchNorm1d(F1),
            # nn.Dropout(dropoutRate)
            nn.Conv1d(F1, F2, kernLength // 2, groups=F1),  # conv 2
        )
        self.stage1 = nn.ModuleList([])
        for _ in range(self.depth[0]):
            self.stage1.append(
                nn.Sequential(
                    nn.Conv1d(F2, F2, kernLength // 2, groups=F2),
                    nn.BatchNorm1d(F2),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                )
            )
        self.stage2 = nn.ModuleList([])
        for _ in range(self.depth[1]):
            self.stage2.append(
                nn.Sequential(
                    SeparableConv1d(F2, F2, kernel_size=31 // 2, padding=31 // 4),  # conv3
                    SeparableConv1d(F2, F2, kernel_size=31 // 2, padding=31 // 4),
                    nn.BatchNorm1d(F2),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                )
            )
        self.norm_s1 = nn.BatchNorm1d(F1)
        self.norm_s2 = nn.BatchNorm1d(F2)

        self.merge_s1 = nn.Sequential(
            nn.AvgPool1d(downSample_1),
            # nn.Conv1d(F1, F2, kernLength // 2, padding=(kernLength // 4), groups=chans)
        )
        self.merge_s2 = nn.Sequential(
            nn.AvgPool1d(downSample_2),
            Permute([0, 2, 1])
        )


        # transformer
        seq_len = int((samples - kernLength + 1) // (downSample_1 * downSample_2))
        embed_dim = F2
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len + 1, embed_dim))

        self.transformer = TransformerVit(dim=embed_dim, depth=6, heads=8, dim_head=256, mlp_dim=128, dropout=0)
        # self.layernorm = nn.LayerNorm(embed_dim)
        self.layernorm = nn.Identity()

        # In: (B, F2 *  (samples - chans + 1) / 32)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, num_classes),
        )

    # @timer_wrap
    def forward(self, x: torch.Tensor):
        # Block 1
        seq_embed = self.forward_embed(x)

        batch_size, seq_len, embed_dim = seq_embed.shape
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, seq_embed), dim=1)
        embeddings += self.pos_embedding
        tr_output = self.transformer(embeddings)
        sequence_output = self.layernorm(tr_output)
        cls_token = sequence_output[:, 0, :]
        y2 = self.classifier(cls_token)

        return y2

    def forward_embed(self, x):
        x = self.stage0(x)
        for stage1 in self.stage1:
            x = stage1(x)
        x = self.merge_s1(x)
        for stage2 in self.stage2:
            x = stage2(x)
        x = self.merge_s2(x)
        return x


class SKAttention1D(nn.Module):

    def __init__(self, channel=512, kernels=[1, 3, 5, 7], reduction=16, group=1, L=32):
        super().__init__()
        self.d = max(L, channel // reduction)
        self.convs = nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv1d(channel, channel, kernel_size=k, padding=k // 2, groups=group)),
                    ('bn', nn.BatchNorm1d(channel)),
                    ('relu', nn.ReLU())
                ]))
            )
        self.fc = nn.Linear(channel, self.d)
        self.fcs = nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d, channel))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        bs, c, _ = x.size()
        conv_outs = []
        ### split
        for conv in self.convs:
            conv_outs.append(conv(x))
        feats = torch.stack(conv_outs, 0)  # k,bs,channel,l

        ### fuse
        U = sum(conv_outs)  # bs,c,l

        ### reduction channel
        S = U.mean(-1)  # bs,c
        Z = self.fc(S)  # bs,d

        ### calculate attention weight
        weights = []
        for fc in self.fcs:
            weight = fc(Z)
            weights.append(weight.view(bs, c, 1))  # bs,channel
        attention_weughts = torch.stack(weights, 0)  # k,bs,channel,1
        attention_weughts = self.softmax(attention_weughts)  # k,bs,channel,1

        ### fuse
        V = (attention_weughts * feats).sum(0)
        return V


class EEGNet_V1x8_Vit_SK(nn.Module):
    def __init__(self, num_classes=2, input_size=(1, 28, 512), dropout_rate=0.5, sampling_rate=128, F1=16, D=6):
        super().__init__()
        Chans: int = input_size[-2]
        Samples: int = input_size[-1]
        kernLength = 31
        F2 = F1 * D

        # Make kernel size and odd number
        try:
            assert kernLength % 2 != 0
        except AssertionError:
            raise ValueError("ERROR: kernLength must be odd number")

        self.seq_embed = nn.Sequential(
            nn.Conv1d(Chans, F1, kernLength, padding=(kernLength // 2)),  # conv1
            nn.BatchNorm1d(F1),
            SKAttention1D(channel=F1, reduction=4),
            nn.Conv1d(F1, F2, Chans, groups=F1),  # conv2
            nn.BatchNorm1d(F2),
            nn.ReLU(),
            SKAttention1D(channel=F2, reduction=4),
            nn.AvgPool1d(4),
            nn.Dropout(dropout_rate),
            SeparableConv1d(F2, F2, kernel_size=31, padding=31 // 2),  # conv3
            nn.BatchNorm1d(F2),
            nn.ReLU(),
            SKAttention1D(channel=F2, reduction=4),
            nn.AvgPool1d(5),
            nn.Dropout(dropout_rate),
            Permute(dims=[0, 2, 1])
        )

        # transformer
        seq_len = (Samples - Chans + 1) // (4 * 5)
        embed_dim = F2
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len + 1, embed_dim))

        self.transformer = TransformerVit(dim=embed_dim, depth=6, heads=8, dim_head=256, mlp_dim=128, dropout=0.)
        # self.layernorm = nn.LayerNorm(embed_dim, eps=1e-12)
        self.layernorm = nn.Identity()

        # In: (B, F2 *  (Samples - Chans + 1) / 32)
        self.classifier = nn.Linear(embed_dim, num_classes)

    # @timer_wrap
    def forward(self, x: torch.Tensor):
        # Block 1
        x = x.squeeze(1)
        seq_embed = self.seq_embed(x)

        batch_size, seq_len, embed_dim = seq_embed.shape
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, seq_embed), dim=1)
        embeddings += self.pos_embedding
        tr_output = self.transformer(embeddings)
        sequence_output = self.layernorm(tr_output)
        cls_token = sequence_output[:, 0, :]
        y2 = self.classifier(cls_token)

        return y2


class EEG_V1x14_Vit_SK_Alab5(nn.Module):
    def __init__(self, num_classes=2, input_size=(1, 28, 512), dropout_rate=0.5, sampling_rate=128, kernLength: int = 31):
        super().__init__()
        chans: int = input_size[-2]
        samples: int = input_size[-1]
        F1 = chans * 2
        F2 = F1 * 2
        downSample_1, downSample_2 = 4, 5
        kernel_2 = 15

        # Make kernel size and odd number
        try:
            assert kernLength % 2 != 0
        except AssertionError:
            raise ValueError("ERROR: kernLength must be odd number")

        self.depth = [1, 2]
        self.stage0 = nn.Sequential(
            # nn.Conv1d(chans, F1, kernLength // 2, padding=(kernLength // 4), groups=chans),  # conv1
            # nn.Conv1d(F1, F2, kernLength // 2, padding=(kernLength // 4), groups=chans),
            # nn.BatchNorm1d(F2),
            # nn.Dropout(dropoutRate)
            nn.Conv1d(chans, F2, kernLength // 2, groups=chans),  # conv 2
        )
        self.stage1 = nn.ModuleList([])
        for _ in range(self.depth[0]):
            self.stage1.append(
                nn.Sequential(
                    nn.Conv1d(F2, F2, kernLength // 2, groups=F2),
                    nn.BatchNorm1d(F2),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                )
            )
        self.stage2 = nn.ModuleList([])
        for _ in range(self.depth[1]):
            self.stage2.append(
                nn.Sequential(
                    SeparableConv1d(F2, F2, kernel_size=31 // 2, padding=31 // 4),  # conv3
                    SeparableConv1d(F2, F2, kernel_size=31 // 2, padding=31 // 4),
                    nn.BatchNorm1d(F2),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                )
            )
        self.norm_s1 = nn.BatchNorm1d(F1)
        self.norm_s2 = nn.BatchNorm1d(F2)

        self.merge_s1 = nn.Sequential(
            nn.AvgPool1d(downSample_1),
            # nn.Conv1d(F1, F2, kernLength // 2, padding=(kernLength // 4), groups=chans)
        )
        self.merge_s2 = nn.Sequential(
            nn.AvgPool1d(downSample_2),
            SKAttention1D(F2, reduction=4),
            Permute([0, 2, 1])
        )


        # transformer
        seq_len = int((samples-chans) // (downSample_1 * downSample_2))
        embed_dim = F2
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len + 1, embed_dim))

        self.transformer = TransformerVit(dim=embed_dim, depth=6, heads=8, dim_head=256, mlp_dim=128, dropout=0.)
        # self.layernorm = nn.LayerNorm(embed_dim)
        self.layernorm = nn.Identity()

        # In: (B, F2 *  (samples - chans + 1) / 32)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, num_classes),
        )

    # @timer_wrap
    def forward(self, x: torch.Tensor):
        # Block 1
        seq_embed = self.forward_embed(x)

        batch_size, seq_len, embed_dim = seq_embed.shape
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, seq_embed), dim=1)
        embeddings += self.pos_embedding
        tr_output = self.transformer(embeddings)
        sequence_output = self.layernorm(tr_output)
        cls_token = sequence_output[:, 0, :]
        y2 = self.classifier(cls_token)

        return y2

    def forward_embed(self, x):
        x = x.squeeze(1)
        x = self.stage0(x)
        for stage1 in self.stage1:
            x = stage1(x)
        x = self.merge_s1(x)
        for stage2 in self.stage2:
            # x_ = x
            # x = stage2(x) + x_
            x = stage2(x)
        x = self.merge_s2(x)
        return x

if __name__ == '__main__':
    x = torch.randn(3, 1, 28, 512)
    net = EEG_V1x14_Vit_SK_Alab5(num_classes=9, input_size=(1, 28, 512))
    print(net(x).shape)