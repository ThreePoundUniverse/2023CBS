import numpy as np
import yaml
from easydict import EasyDict
import os
from scipy.stats import norm, mstats, kurtosis, skew, theilslopes
from sklearn.model_selection import KFold

feature = False

with open('./config.yaml') as f:
    CFG = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

def split_window(data_trial, label_trial, windowLength, windowStep):

    # 取10s窗口，无重叠
    data_trial = data_trial[:, :int(data_trial.shape[1] / CFG.srate / windowStep) * CFG.srate * windowStep]
    sample = np.arange(windowLength, data_trial.shape[1] / CFG.srate + 1e-8, windowStep)
    sample = (sample * CFG.srate).astype(np.int64)
    data_tmp = np.zeros((len(sample), data_trial.shape[0], windowLength * CFG.srate), dtype=np.float32)
    for data_i in range(len(sample)):
        data_tmp[data_i, :, :] = data_trial[:, (sample[data_i] - windowLength * CFG.srate):sample[data_i]]
        # io.savemat("input_data", {'name':data_tmp[1, :, :].T})



    return data_tmp, np.array([label_trial] * len(sample))

def FWHM(x, y):
    half_max = max(y) / 2.
    # find when function crosses line half_max (when sign of diff flips)
    # take the 'derivative' of signum(half_max - y[])
    d = np.sign(half_max - np.array(y[0:-1])) - np.sign(half_max - np.array(y[1:]))
    # find the left and right most indexes
    left_idx = np.where(d > 0)[0][0]
    right_idx = np.where(d < 0)[0][-1]
    return x[right_idx] - x[left_idx] # return the difference (full width)

def extract_feature(data, alpha=0.05):

    kur = kurtosis(data, axis=-1)
    skewness = skew(data, axis=-1)
    std = np.std(data, axis=-1)
    mean = np.mean(data, axis=-1)
    max_ = np.max(data, axis=-1)
    min = np.min(data, axis=-1)
    feature_com = np.stack((kur, skewness, std, mean, max_, min), axis=1)
    return feature_com

def getdata(subject):
    data_train = np.empty([0, 40, CFG.windowLength*CFG.srate])
    label_train = np.empty([0])
    data_valid = np.empty([0, 40, CFG.windowLength * CFG.srate])
    label_valid = np.empty([0])

    for i in os.listdir("process_data/{}".format(CFG.folder)):
        path = os.path.join("process_data/{}".format(CFG.folder), i)
        data = np.load(path, allow_pickle=True)

        if i != subject:
            for trial in range(24):
                data_trial, label_trial = data[trial, 0], data[trial, 1]
                # negvspos
                # if label_trial == 1:
                #     continue
                # elif label_trial == 2:
                #     label_trial = 1
                # negvsneu
                # if label_trial == 2:
                #     continue
                # posvsneu
                # if label_trial == 0:
                #     continue
                # elif label_trial == 1:
                #     label_trial = 0
                # elif label_trial == 2:
                #     label_trial = 1

                data_trial = data_trial[:, -40 * CFG.srate:]

                data_, label_ = split_window(data_trial, label_trial, CFG.windowLength, CFG.windowStep)
                # 取刺激开始后10s数据

                data_train = np.vstack([data_train, data_])
                label_train = np.append(label_train, label_)
        elif i == subject:
            for trial in range(24):
                data_trial, label_trial = data[trial, 0], data[trial, 1]
                # negvspos
                # if label_trial == 1:
                #     continue
                # elif label_trial == 2:
                #     label_trial = 1
                # # negvsneu
                # if label_trial == 2:
                #     continue
                # posvsneu
                # if label_trial == 0:
                #     continue
                # elif label_trial == 1:
                #     label_trial = 0
                # elif label_trial == 2:
                #     label_trial = 1

                data_trial = data_trial[:, -40 * CFG.srate:]


                data_, label_ = split_window(data_trial, label_trial, CFG.windowLength, CFG.windowStep)

                # 取刺激开始后10s数据
                data_valid = np.vstack([data_valid, data_])
                label_valid = np.append(label_valid, label_)

    if feature:
        data_train = np.mean(data_train, axis=-1)
        data_valid = np.mean(data_valid, axis=-1)
    return data_train, label_train, data_valid, label_valid


# if __name__=='__main__':
#     import random
#
#     data, label = getdata()
#     kf = KFold(n_splits=5, shuffle=True, random_state=666)
#     for train_index, test_index in kf.split(data):
#         data_test = np.empty([0, CFG.channels, CFG.windowLength * CFG.srate])
#         y_test = np.array([])
#         data_train = np.empty([0, CFG.channels, CFG.windowLength * CFG.srate])
#         y_train = np.array([])
#
#         for trial in test_index:
#             data_test_trial, y_test_trial = data[trial], label[trial]
#             data_test = np.vstack([data_test, data_test_trial])
#             y_test = np.append(y_test, y_test_trial)
#         for trial in train_index:
#             data_train_trial, y_train_trial = data[trial], label[trial]
#             data_train = np.vstack([data_train, data_train_trial])
#             y_train = np.append(y_train, y_train_trial)
#
#         print(data_train.shape, y_train.shape, data_test.shape, y_test.shape)
















