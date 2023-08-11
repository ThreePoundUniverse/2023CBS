# This is the networks script
import torch
import torch.nn as nn
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

channel_naoye = (np.array([1,2,3,4,5,6,7,8,9,10,11,12,14,15,13,17,18,21,23,24,27,29,30,33,35,36,39,16,19,20,22,25,26,28,31,32,34,37,38,40]) - 1).tolist()
channel_banqiu = (np.array([1,2,5,6,8,10,14,13,17,18,21,23,24,27,29,30,33,35,36,39,4,3,7,9,11,12,15,16,20,19,22,26,25,28,32,31,34,38,37,40]) - 1).tolist()
channel_1 = (np.array([1,2,3,4,5,6,7,8,9,10,11,12,14,15]) - 1).tolist()
channel_2 = (np.array([13,17,18,21,23,24,27,29,30,33,35,36,39]) - 1).tolist()
channel_3 = (np.array([16,19,20,22,25,26,28,31,32,34,37,38,40]) - 1).tolist()

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
        return x * y.expand_as(x), weight

# class TSception(nn.Module):
#     def conv_block(self, in_chan, out_chan, kernel, step, pool):
#         return nn.Sequential(
#             nn.Conv2d(in_channels=in_chan, out_channels=out_chan,
#                       kernel_size=kernel, stride=step),
#             nn.LeakyReLU(),
#             nn.AvgPool2d(kernel_size=(1, pool), stride=(1, pool)))
#
#     def __init__(self, num_classes, input_size, sampling_rate, num_T, num_S, hidden, dropout_rate):
#         # input_size: 1 x EEG channel x datapoint
#         super(TSception, self).__init__()
#         self.inception_window = [0.5, 0.25, 0.125]
#         self.pool = 8
#         # by setting the convolutional kernel being (1,lenght) and the strids being 1 we can use conv2d to
#         # achieve the 1d convolution operation
#         self.Tception1 = self.conv_block(1, num_T, (1, int(self.inception_window[0] * sampling_rate)), 1, self.pool)
#         self.Tception2 = self.conv_block(1, num_T, (1, int(self.inception_window[1] * sampling_rate)), 1, self.pool)
#         self.Tception3 = self.conv_block(1, num_T, (1, int(self.inception_window[2] * sampling_rate)), 1, self.pool)
#
#         self.Sception1 = self.conv_block(num_T, num_S, (int(input_size[1]), 1), 1, int(self.pool*0.25))
#         self.Sception2 = self.conv_block(num_T, num_S, (int(input_size[1] * 0.5), 1), (int(input_size[1] * 0.5), 1),
#                                          int(self.pool*0.25))
#         self.fusion_layer = self.conv_block(num_S, num_S, (3, 1), 1, 4)
#         self.BN_t = nn.BatchNorm2d(num_T)
#         self.BN_s = nn.BatchNorm2d(num_S)
#         self.BN_fusion = nn.BatchNorm2d(num_S)
#
#         self.fc = nn.Sequential(
#             nn.Linear(num_S, hidden),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),
#             nn.Linear(hidden, num_classes)
#         )
#
#     def forward(self, x):
#         x = x.unsqueeze(1)
#         y = self.Tception1(x)
#         out = y
#         y = self.Tception2(x)
#         out = torch.cat((out, y), dim=-1)
#         y = self.Tception3(x)
#         out = torch.cat((out, y), dim=-1)
#         out = self.BN_t(out)
#         z = self.Sception1(out)
#         out_ = z
#         z = self.Sception2(out[:, :, channel_banqiu, :])
#         out_ = torch.cat((out_, z), dim=2)
#         out = self.BN_s(out_)
#         out = self.fusion_layer(out)
#         out = self.BN_fusion(out)
#         out = torch.squeeze(torch.mean(out, dim=-1), dim=-1)
#         out = self.fc(out)
#         return out


class TSception(nn.Module):
    def conv_block(self, in_chan, out_chan, kernel, step, pool):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=out_chan,
                      kernel_size=kernel, stride=step),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(1, pool), stride=(1, pool)))
    def conv_block_naoye(self, in_chan, out_chan, kernel, step, pool):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=out_chan,
                      kernel_size=kernel, stride=step, padding=(1, 0)),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(1, pool), stride=(1, pool)))

    def __init__(self, num_classes, input_size, sampling_rate, num_T, num_S, hidden, dropout_rate):
        # input_size: 1 x EEG channel x datapoint
        super(TSception, self).__init__()
        self.inception_window = [0.5, 0.25, 0.125]
        self.pool = 8
        self.channel_naoye = channel_naoye
        self.channel_banqiu = channel_banqiu
        # by setting the convolutional kernel being (1,lenght) and the strids being 1 we can use conv2d to
        # achieve the 1d convolution operation
        self.Tception1 = self.conv_block(1, num_T, (1, int(self.inception_window[0] * sampling_rate)), 1, self.pool)
        self.Tception2 = self.conv_block(1, num_T, (1, int(self.inception_window[1] * sampling_rate)), 1, self.pool)
        self.Tception3 = self.conv_block(1, num_T, (1, int(self.inception_window[2] * sampling_rate)), 1, self.pool)

        self.SE_Block = SE_Block(ch_in=40, reduction=4)

        self.Sception1 = self.conv_block(num_T, num_S, (int(input_size[1]), 1), 1, 1)
        self.Sception2 = self.conv_block(num_T, num_S, (int(input_size[1] * 0.5), 1), (int(input_size[1] * 0.5), 1), 1)
        self.Sception3 = self.conv_block(num_T, num_S, (14, 1), (14, 1), 1)
        self.fusion_layer = nn.Sequential(
            self.conv_block(num_S, num_S, (3, 1), 1, 1),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.BN_t = nn.BatchNorm2d(num_T)
        self.BN_s = nn.BatchNorm2d(num_S)
        self.BN_fusion = nn.BatchNorm2d(num_S)
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(num_S, hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        y = self.Tception1(x)
        out = y
        y = self.Tception2(x)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(x)
        out = torch.cat((out, y), dim=-1) # [16, 15, 40, 5]
        # out = self.BN_t(out)
        bs, ks, ch, t = out.size()
        # 通道注意力
        # out_atten = out.permute(0, 2, 1, 3).reshape(bs, ch, -1)
        # out_atten, weight = self.SE_Block(out_atten)
        # out = out_atten.reshape(bs, ch, ks, t).permute(0, 2, 1, 3)
        #全局
        z1 = self.Sception1(out)
        #半球
        z2 = self.Sception2(out[:, :, self.channel_banqiu, :])
        #脑叶
        out_naoye = out[:, :, self.channel_naoye, :]
        zero_channel = torch.zeros([out.shape[0], out.shape[1], 42, out.shape[3]]).to(device) #[16, 15, 42, 5]
        zero_channel[:, :, :27, :] = out_naoye[:, :, :27, :]
        zero_channel[:, :, 28:-1, :] = out_naoye[:, :, 27:, :]
        z3 = self.Sception3(zero_channel)
        out = torch.cat([z1, z2], dim=2)
        out = self.BN_s(out)
        out = self.fusion_layer(out)
        out = self.BN_fusion(out)
        # out = self.fc(torch.cat([featrue, self.flatten(out)], dim=-1))
        out = self.fc(self.flatten(out))
        return out
#
#
# class ANN(nn.Module):
#     def __init__(self, num_classes, channel):
#         super(ANN, self).__init__()
#         self.block = nn.Sequential(
#             nn.Linear(40, 80),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(80, num_classes)
#         )
#
#     def forward(self, x):
#
#         x = torch.mean(x, dim=-1)
#         out = self.block(x)
#         return out

if __name__ == '__main__':
    with torch.no_grad():
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        cuda0 = torch.device('cuda:0')
        x = torch.rand((16, 40, 160), device=cuda0)
        model = TSception(num_classes=3, input_size=(1, 40, 160), sampling_rate=160, num_T=15, num_S=15, hidden=32, dropout_rate=0.5)
        # model = ANN(num_classes=3, channel=40)
        model.cuda()
        output= model(x)
        print('output:', output.shape)