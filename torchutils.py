import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold
from copy import deepcopy
import numpy as np
import os
import os.path as osp
from tqdm import tqdm
from time import strftime, localtime
import pandas as pd
from pathlib import Path
from easydict import EasyDict
import yaml

from spikingjelly.activation_based import rnn, encoding, layer, neuron, surrogate, functional
# # 指定seed
# m_seed = 3407
#
# # 设置seed
# torch.manual_seed(m_seed)
# torch.cuda.manual_seed_all(m_seed)
with open('./config.yaml') as f:
    CFG = EasyDict(yaml.load(f, Loader=yaml.FullLoader))


def contrastLoss(representations, label):
    T = 0.5  # 温度参数T
    n = label.shape[0]  # batch
    # 这步得到它的相似度矩阵
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
    # 这步得到它的label矩阵，相同label的位置为1
    mask = torch.ones_like(similarity_matrix) * (label.expand(n, n).eq(label.expand(n, n).t()))

    # 这步得到它的不同类的矩阵，不同类的位置为1
    mask_no_sim = torch.ones_like(mask) - mask

    # 这步产生一个对角线全为0的，其他位置为1的矩阵
    mask_dui_jiao_0 = torch.ones(n, n) - torch.eye(n, n)

    # 这步给相似度矩阵求exp,并且除以温度参数T
    similarity_matrix = torch.exp(similarity_matrix / T)

    # 这步将相似度矩阵的对角线上的值全置0，因为对比损失不需要自己与自己的相似度
    similarity_matrix = similarity_matrix * mask_dui_jiao_0

    # 这步产生了相同类别的相似度矩阵，标签相同的位置保存它们的相似度，其他位置都是0，对角线上也为0
    sim = mask * similarity_matrix

    # 用原先的对角线为0的相似度矩阵减去相同类别的相似度矩阵就是不同类别的相似度矩阵
    no_sim = similarity_matrix - sim

    # 把不同类别的相似度矩阵按行求和，得到的是对比损失的分母(还差一个与分子相同的那个相似度，后面会加上)
    no_sim_sum = torch.sum(no_sim, dim=1)

    '''
    将上面的矩阵扩展一下，再转置，加到sim（也就是相同标签的矩阵上），然后再把sim矩阵与sim_num矩阵做除法。
    至于为什么这么做，就是因为对比损失的分母存在一个同类别的相似度，就是分子的数据。做了除法之后，就能得到
    每个标签相同的相似度与它不同标签的相似度的值，它们在一个矩阵（loss矩阵）中。
    '''
    no_sim_sum_expend = no_sim_sum.repeat(n, 1).T
    sim_sum = sim + no_sim_sum_expend
    loss = torch.div(sim, sim_sum)

    '''
    由于loss矩阵中，存在0数值，那么在求-log的时候会出错。这时候，我们就将loss矩阵里面为0的地方
    全部加上1，然后再去求loss矩阵的值，那么-log1 = 0 ，就是我们想要的。
    '''
    loss = mask_no_sim + loss + torch.eye(n, n)

    # 接下来就是算一个批次中的loss了
    loss = -torch.log(loss)  # 求-log
    # loss = torch.sum(torch.sum(loss, dim=1)) / (2 * n)  # 将所有数据都加起来除以2n

    # 最后一步也可以写为---建议用这个， (len(torch.nonzero(loss)))表示一个批次中样本对个数的一半
    loss = torch.sum(torch.sum(loss, dim=1)) / (len(torch.nonzero(loss)))

    return loss

class TorchDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.len = len(data)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        data = torch.from_numpy(data).float()
        # data = torch.from_numpy(data).type(torch.complex64)
        label = torch.tensor(label).long()
        return data, label


class CrossVal:
    def __init__(self, train_data, train_label, val_data, val_label, model, config):
        self.config = config
        self.model = model
        self.optimizer = Adam(model.parameters(), lr=self.config.lr, weight_decay=self.config.l2norm)
        self.loss_func = nn.CrossEntropyLoss()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.best_state = {'loss': 1e8}
        self.best_state = {'acc': 0.0}
        train_dataset = TorchDataset(train_data, train_label)
        val_dataset = TorchDataset(val_data, val_label)
        self.train_loader = DataLoader(train_dataset, batch_size=self.config.train_bs, shuffle=True, num_workers=0, drop_last=False)
        self.val_loader = DataLoader(val_dataset, batch_size=self.config.val_bs, shuffle=False, num_workers=0, drop_last=False)
        self.history = {'acc': [], 'loss': [], 'val_acc': [], 'val_loss': []}
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                    mode='max', factor=0.1,
                                                                    patience=self.config.patience * 0.7,
                                                                    verbose=False, min_lr=1e-7)
        self.earlystop_callback = EarlyStopping(mode='max', patience=self.config.patience, min_delta=0, verbose=False)

    def init_model_n_optimizer(self):
        def weight_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                # nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.model.apply(weight_init)
        self.optimizer.state = collections.defaultdict(dict)

    def check_path(self, path):
        pl_path = Path(path)
        if pl_path.suffix == '':
            if not osp.exists(path):
                os.makedirs(path)
        else:
            if not osp.exists(pl_path.parent):
                os.makedirs(pl_path.parent)
        return path

    def to_device(self, data, label):
        self.model.to(self.device)
        self.loss_func.to(self.device)
        data_ = data.to(self.device)
        label_ = label.to(self.device)
        return data_, label_

    def save_ckpt(self, state):
        self.check_path(self.config.ckpt_dir)
        filename = osp.join(self.config.ckpt_dir, self.config.ckpt_name)
        torch.save(state, filename + ".pth.tar")

    def load_ckpt(self, path=None):
        if not path:
            path = osp.join(self.config.ckpt_dir, self.config.ckpt_name) + ".pth.tar"
        if osp.isfile(path):
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint["state_dict"], strict=False)

    def val(self):
        total_num, total_loss = 0, 0.0
        total_correct = 0.0

        self.model.eval()
        with torch.no_grad():
            for b_index, (data, label) in enumerate(self.val_loader):
                data, label = self.to_device(data, label)
                out = self.model(data)
                loss_ = self.loss_func(out, label)

                total_num += data.size(0)
                total_loss += loss_.item() * data.size(0)

                prediction = torch.argmax(out, dim=-1)
                total_correct += prediction.eq(label).sum().item()

        loss = total_loss / total_num
        acc = (1.0*total_correct) / (total_num * 1.0)

        return loss, acc

    def package_state(self, loss=None, acc=None):
        state = {"state_dict": self.model.state_dict()}
        if loss and 'loss' in self.best_state.keys():
            if loss < self.best_state['loss']:
                self.save_ckpt(state)
                self.best_state['loss'] = loss
                return
        if acc and 'acc' in self.best_state.keys():
            if acc >= self.best_state['acc']:
                self.save_ckpt(state)
                self.best_state['acc'] = acc
                return

    def log(self, tr_acc, tr_loss, val_acc, val_loss):
        # history_ = deepcopy(self.history)

        self.history['acc'].append(tr_acc)
        self.history['loss'].append(tr_loss)
        self.history['val_acc'].append(val_acc)
        self.history['val_loss'].append(val_loss)

    def fit(self):
        self.init_model_n_optimizer()
        # self.model.train()
        with torch.enable_grad():
            for ep_index in range(self.config.epochs):
                total_num, total_loss = 0, 0.0
                total_correct = 0.0
                train_bar = tqdm(enumerate(self.train_loader),
                                 desc=f'[{strftime("%Y/%m/%d-%H:%M:%S")}] TrainEpoch {ep_index + 1}',
                                 ascii=True, ncols=100)
                self.model.train()

                for b_index, (data, label) in train_bar:
                    data, label = self.to_device(data, label)
                    out = self.model(data)
                    loss_ = self.loss_func(out, label)

                    self.optimizer.zero_grad()
                    loss_.backward()
                    self.optimizer.step()

                    batch_size = data.size(0)
                    total_num += batch_size
                    total_loss += loss_.item() * batch_size

                    prediction = torch.argmax(out, dim=-1)
                    correct_ = prediction.eq(label).sum().item()
                    total_correct += correct_

                    loss_step = loss_.item()
                    acc_step = correct_ / batch_size

                    train_bar.set_postfix_str(f'tr loss:{loss_step}, tr acc:{acc_step}')

                val_loss, val_acc = self.val()
                self.package_state(loss=val_loss, acc=val_acc)

                loss = total_loss / total_num
                acc = total_correct / total_num

                self.log(acc, loss, val_acc, val_loss)

                if self.scheduler is not None:
                    self.scheduler.step(val_acc)

                self.earlystop_callback(metric=val_acc)
                if self.earlystop_callback.early_stop is True:
                    break
                print("epoch:{} val_acc:{:.5f} val_loss:{}".format(ep_index+1, val_acc, val_loss))
        # print("val datasets best loss:", self.best_state['loss'])
        # file = open('result.txt', mode='a')
        # file.write(str(self.best_state['acc']) + '\n')
        # file.close()

    def predict(self, data):
        pseudo_label = np.zeros(data.shape[0], dtype=np.int64)
        loader = DataLoader(TorchDataset(data, pseudo_label),
                            batch_size=2, shuffle=False, num_workers=0)
        self.model.eval()
        preds = []
        # weight = np.empty([0, 40])
        with torch.no_grad():
            for b_index, (data, label) in enumerate(loader):
                data, _ = self.to_device(data, label)
                out = self.model(data)
                # weight = np.vstack((weight, w.cpu().numpy()))
                prediction = torch.argmax(out, dim=-1)
                preds.append(prediction.cpu().numpy())
        return np.concatenate(preds)

class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    def __init__(self, mode='min', patience=5, min_delta=0, verbose=False):
        """
        :param patience: how many epochs to wait before stopping when loss/acc is
               not improving
        :param min_delta: minimum difference between new loss/acc and old loss/acc for
               new loss/acc to be considered as an improvement
        """
        assert mode in {'min', 'max', None}
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_metric = None
        self.early_stop = False

    def __call__(self, metric):
        if self.mode == 'min':
            self._min_check(min_check_val=metric)
        elif self.mode == 'max':
            self._max_check(max_check_val=metric)
        else:
            raise NotImplementedError

    def _min_check(self, min_check_val):
        if self.best_metric is None:
            self.best_metric = min_check_val
        elif self.best_metric - min_check_val > self.min_delta:
            self.best_metric = min_check_val
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_metric - min_check_val < self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                if self.verbose:
                    print('INFO: Early stopping')
                self.early_stop = True

    def _max_check(self, max_check_val):
        if self.best_metric is None:
            self.best_metric = max_check_val
        elif max_check_val - self.best_metric > self.min_delta:
            self.best_metric = max_check_val
            # reset counter if validation loss improves
            self.counter = 0
        elif max_check_val - self.best_metric < self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                if self.verbose:
                    print('INFO: Early stopping')
                self.early_stop = True
