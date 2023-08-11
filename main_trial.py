# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 15:37:00 2022
@author: yulin sun
Contact: syuri@tju.edu.cn
"""

# %% Import
import os
import gc
import numpy as np
from read_NIRS_cs import getdata
from sklearn.model_selection import KFold
from torchutils import CrossVal
from tsception_new import TSception
from TIE_EEGNet_en import Tie_EEGNet
from my_model import DBJNet, mean, fnirsNet_spatial, CNN_NLSTM
import yaml
import torch
# from network import TSception
import random
from easydict import EasyDict
from sklearn.metrics import roc_curve,precision_recall_curve ,roc_auc_score,auc,average_precision_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# %% Configuration
# %% Configuration
with open('./config.yaml') as f:
    CFG = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

def seed_torch(seed=666):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch()




if __name__=='__main__':
    import random

    f = open("result_aug/result.txt", 'w', encoding='utf-8')
    acc = []
    f1 = []
    index = 0
    y_true = []
    y_pred = []
    for i in os.listdir("process_data/{}".format(CFG.folder)):
        name, _ = i.split('.')
        data_train, y_train, data_test, y_test = getdata(i)
        print(data_train.shape, y_train.shape, data_test.shape, y_test.shape) # (216, 40, 800) (216,) (24, 40, 800) (24,)


        data_train, data_valid, y_train, y_valid = train_test_split(data_train, y_train, test_size=0.2, random_state=42)


        print(data_train.shape, y_train.shape, data_valid.shape, y_valid.shape, data_test.shape, y_test.shape)
        index+=1
        # save model name
        CFG.ckpt_dir = f'./ckpt/{CFG.model_name}'
        CFG.ckpt_name = f'ckpt_{index}'

        # Model
        model = DBJNet(n_class=CFG.num_classes, channels=CFG.channels, samples=CFG.srate * CFG.windowLength)
        # model = TSception(num_classes=CFG.num_classes, input_size=(1, CFG.channels, CFG.srate * CFG.windowLength),
        #                   sampling_rate=CFG.srate * CFG.windowLength, num_T=15, num_S=15, hidden=32, dropout_rate=0.5)
        # model = Tie_EEGNet(n_class=CFG.num_classes, channels=CFG.channels, samples=CFG.srate * CFG.windowLength)
        # model = CNN_NLSTM(n_class=CFG.num_classes, channels=CFG.channels, samples=CFG.srate * CFG.windowLength)
        cv_obj = CrossVal(data_train, y_train, data_valid, y_valid, model, CFG)
        cv_obj.fit()
        history = cv_obj.history

        gc.collect()
        cv_obj.load_ckpt()
        label_pred = cv_obj.predict(data_test)
        # label_pred_arg = np.argmax(label_pred, axis=1)
        test_acc = accuracy_score(y_test, label_pred)

        y_true = y_true + list(y_test)
        y_pred = y_pred + list(label_pred)
        if CFG.num_classes == 3:
            # best_f1 = f1_score(y_test, label_pred, average='macro')
            best_f1 = f1_score(y_test, label_pred, average='weighted')
        else:
            best_f1 = f1_score(y_test, label_pred, average='weighted')
        del data_train
        del data_test
        gc.collect()
        #############################


        acc.append(test_acc)
        f1.append(best_f1)
        print("subject:{}, test_acc:{:.5f}, f1:{:.5f}".format(i, test_acc, best_f1))
        name, _ = i.split('.')
        f.writelines(name + "\t" + str(test_acc) + "\t" + str(best_f1) + '\n')
        f.flush()
    np.save("y_true.npy", np.array(y_true))
    np.save("y_pred.npy", np.array(y_pred))
    print(acc)
    print(f1)
    print(np.mean(acc), np.std(acc), np.mean(f1), np.std(f1))
    f.writelines("mean" + "\t" + str(np.mean(acc)) + "\t" + str(np.std(acc)) + "\t" + str(np.mean(f1)) + "\t" + str(np.std(f1)) + '\n')
    f.close()




























































