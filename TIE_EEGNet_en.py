import torch.nn as nn
from TIE_Layer_en import TIE_Layer
import torch
import numpy as np

channel_seq = (np.array([1,2,3,4,5,6,7,8,9,10,11,12,14,15,13,17,18,21,23,24,27,29,30,33,35,36,39,16,20,19,22,26,25,28,32,31,34,38,37,40])-1).tolist()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # 指定seed
# m_seed = 3407
#
# # 设置seed
# torch.manual_seed(m_seed)
# torch.cuda.manual_seed_all(m_seed)

class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm = 1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(x)

class Tie_EEGNet(nn.Module):
    def CalculateOutSize(self, model, channels, samples):
        data = torch.rand(1,1,channels, samples)
        model.eval()
        out = model(data).shape
        return out[2:]

    def ClassifierBlock(self, inputSize, n_class):
        return nn.Sequential(
            nn.Linear(inputSize, n_class, bias= False),
            nn.Softmax(dim = 1)
        )

    def __init__(self, n_class = 3, channels = 40, samples = 160, kernel_length = 80, kernel_length2 = 20, alpha=2, dropoutRate = 0.5,
                 F1 = 8, F2 = 16, D = 2,
                 tie = 'sinusoidal', isTrain= True, pool='Avg'):
        super(Tie_EEGNet, self).__init__()
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.samples = samples
        self.n_class = n_class
        self.channels = channels
        self.dropoutRate = dropoutRate
        self.kernel_length = kernel_length
        self.kernel_length2 = kernel_length2
        self.tie = tie
        self.isTrain = isTrain
        self.alpha=alpha
        self.pool = pool

        # self.Conv2d_1 = nn.Conv2d(1, self.F1, (1, self.kernel_length), padding=(0, self.kernel_length // 2), bias = False)#'same'
        self.Conv2d_1 = nn.Conv2d(1, self.F1, (1, self.kernel_length), padding='same', bias=False)  # 'same'

        # self.TIE_Layer = TIE_Layer(pool= self.pool, tie= self.tie, conv2Doutput= self.Conv2d_1, inc = 1, outc = self.F1,
        #                           kernel_size = (1, self.kernel_length), pad=(0, self.kernel_length // 2), stride = 1, bias = False,
        #                           sample_len= self.samples,is_Train=self.isTrain,alpha=self.alpha)
        self.BatchNorm_1_1 = nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps = 1e-3)
        self.Depthwise_Conv2d = Conv2dWithConstraint(self.F1, self.F1 * self.D, (self.channels, 1), stride=1, max_norm= 1, groups= self.F1, bias = False) #, padding='valid'

        self.BatchNorm_1_2 = nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps = 1e-3)
        self.elu1 = nn.ELU(inplace=True)
        self.avg_pool_1 = nn.AvgPool2d((1, 4), stride= 4)
        self.Dropout_1 = nn.Dropout(p= self.dropoutRate)
        self.Separable_Conv2d_1 = nn.Conv2d(self.F1 * self.D, self.F1 * self.D, (1, self.kernel_length2), padding=(0, self.kernel_length // 2), bias= False, groups= self.F1 * self.D) # 'same'
        self.Separable_Conv2d_2 = nn.Conv2d(self.F1 * self.D, self.F2, 1, padding= (0, 0), bias= False, groups= 1)
        self.BatchNorm_2 = nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps = 1e-3)
        self.elu2 = nn.ELU(inplace=True)
        self.avg_pool_2 = nn.AvgPool2d((1, 8), stride= 8)
        self.Dropout_2 = nn.Dropout(p= self.dropoutRate)


        self.fea_model = nn.Sequential(self.Conv2d_1,
                                       self.BatchNorm_1_1,
                                       self.Depthwise_Conv2d,
                                       self.BatchNorm_1_2,
                                       nn.ELU(inplace=True),
                                       self.avg_pool_1,
                                       self.Dropout_1,
                                       self.Separable_Conv2d_1,
                                       self.Separable_Conv2d_2,
                                       self.BatchNorm_2,
                                       nn.ELU(inplace=True),
                                       self.avg_pool_2,
                                       self.Dropout_2)

        self.fea_out_size = self.CalculateOutSize(self.fea_model, self.channels, self.samples)
        self.classifierBlock = self.ClassifierBlock(self.F2 * self.fea_out_size[1], self.n_class)

    def forward(self, data):
        conv_data = self.fea_model(data.unsqueeze(1))

        # conv_data = self.Conv2d_1(data.unsqueeze(1))
        # conv_data = self.BatchNorm_1_1(conv_data)
        # conv_data = self.Depthwise_Conv2d(conv_data)
        # conv_data = self.BatchNorm_1_2(conv_data)
        # conv_data = self.elu1(conv_data)
        # conv_data = self.avg_pool_1(conv_data)
        # conv_data = self.Dropout_1(conv_data)
        # conv_data = self.Separable_Conv2d_1(conv_data)
        # conv_data = self.Separable_Conv2d_2(conv_data)
        # conv_data = self.BatchNorm_2(conv_data)
        # conv_data = self.elu2(conv_data)
        # conv_data = self.avg_pool_2(conv_data)
        # conv_data = self.Dropout_2(conv_data)

        flatten_data = conv_data.view(conv_data.size()[0], -1) # [16, 240]
        pred_label = self.classifierBlock(flatten_data)

        return pred_label


if __name__ == '__main__':
    with torch.no_grad():
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        cuda0 = torch.device('cuda:0')
        x = torch.rand((16, 60, 600), device=cuda0)
        model = Tie_EEGNet(n_class=3, channels=60, samples=600)
        model.cuda()
        output = model(x)
        print('output:', output.shape)


