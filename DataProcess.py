from scipy.io import loadmat
import scipy.io as scio
import numpy as np
import os
from scipy import signal
from scipy.fftpack import dct, idct

channel_left = (np.array([13,17,18,21,23,24,27,29,30,33,35,36,39]) - 1).tolist()
channel_right = (np.array([16,20,19,22,26,25,28,32,31,34,38,37,40]) - 1).tolist()

def filter_bp(x,fs,wl,wh):
    fN = 3
    fc = fs/2
    w1c = wl/fc
    w2c = wh/fc
    b, a = signal.butter(fN, [w1c, w2c],'bandpass')
    x_filter = signal.filtfilt(b,a,x)
    return x_filter


gap_pos = []
gap_neg = []
for i in os.listdir(r"D:\言语想象（JNE数据）\try\4.preproccData\data_out_task1_0.5"):
    name, _ = i.split(".")
    if name == "20220720lijiayi_20220720_122025_1":
        continue

    ########获得被试的onset
    data_task1 = loadmat(r"D:\言语想象（JNE数据）\try\4.preproccData\data_out_task1_0.5\{}.mat".format(name))
    data_task2 = loadmat(r"D:\言语想象（JNE数据）\try\4.preproccData\data_out_task2_0.5\{}.mat".format(name))

    oxyData_task1 = data_task1['dcNbf'][:, 0, :].T # (40, 82705)
    oxyData_task2 = data_task2['dcNbf'][:, 0, :].T  # (40, 82705)
    sample = int(data_task1['fs'][0][0])
    # #
    if sample != 4:
        continue
    oxyData = np.append(oxyData_task1, oxyData_task2, axis=1)
    print(oxyData.shape)

    # oxyData = filter_bp(oxyData, fs=sample, wl=0.01, wh=1.99)
    #
    # channel_change = [13,17,18,21,23,24,27,29,30,33,35,36,39,1,2,5,8,10,14,6,11,3,4,7,9,12,15,16,19,20,22,25,26,28,31,32,34,37,38,40]
    # channel_change = [1,2,5,6,8,10,14,13,17,18,21,23,24,27,29,30,33,35,36,39,4,3,7,11,9,12,15,16,20,19,22,26,25,28,32,31,34,38,37,40]
    # for i in range(len(channel_change)):
    #     channel_change[i] = channel_change[i]-1

    # oxyData = oxyData[channel_change, :]
    onset_task1 = list(data_task1['s'].squeeze())
    onset_task2 = list(data_task2['s'].squeeze())
    onset = onset_task1 + onset_task2
    #
    index = 0

    all_trails = []
    for i in range(len(onset)):
        if onset[i] == 33:
            start = i
            label = 2
        elif onset[i] == 44:
            start = i
            label = 1
        elif onset[i] == 55:
            start = i
            label = 0
        elif onset[i] == 7:
            single_trail = []
            index += 1
            end = i

            ### 基线校正---减去刺激呈现前2s的平均响应
            if label != 1:
                oxy_data = np.zeros_like(oxyData[:, start:end], dtype=np.float32)
                for j in range(oxyData.shape[0]):
                    oxy_data[j, :] = oxyData[j, start:end] - np.mean(oxyData[j, start - 5 * sample:start])
            else:
                oxy_data = np.zeros_like(oxyData[:, start:start+40*sample], dtype=np.float32)
                for j in range(oxyData.shape[0]):
                    oxy_data[j, :] = oxyData[j, start:start+40*sample] - np.mean(oxyData[j, start - 5 * sample:start])

            # oxy_data = np.zeros_like(oxyData[:, start:end], dtype=np.float32)
            # for j in range(oxyData.shape[0]):
            #     oxy_data[j, :] = oxyData[j, start:end] - np.mean(oxyData[j, start-2*sample:start])
            # if sample > 4:
            #     oxy_data = signal.resample_poly(oxy_data.T, 4, sample).T
            scale_mean = np.mean(oxy_data, axis=1)
            scale_std = np.std(oxy_data, axis=1)
            oxy_data = ((oxy_data.T - scale_mean) / scale_std).T

            single_trail.append(oxy_data)
            single_trail.append(label)
            all_trails.append(np.array(single_trail))
        else:
            continue
    np.save("process_data/4hz/{}.npy".format(name), np.array(all_trails))
#
# data = np.load("process_data/4hz_nolvbo/20220720lijiayi_20220720_122025_1.npy", allow_pickle=True)
#
# for i in range(data.shape[0]):
#     oxy_data, label = data[i, 0], data[i, 1]
#     print(oxy_data.shape, label)


