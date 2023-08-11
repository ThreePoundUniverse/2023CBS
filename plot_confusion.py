import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, labels_name, title, colorbar=True, cmap=None):
    fs = 15
    plt.imshow(cm, interpolation='nearest', cmap='Reds')    # 在特定的窗口上显示图像
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.annotate(cm[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')
    if colorbar:
        plt.colorbar(plt.clim(0,100))
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name, fontsize=fs)    # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name, fontsize=fs)    # 将标签印在y轴坐标上
    plt.title(title)    # 图像标题
    # plt.ylabel('True label', fontsize=fs)
    # plt.xlabel('Predicted label', fontsize=fs)
    plt.savefig("confusion.svg",format='svg')
    #plt.show()


y_true = np.load("y_true.npy")
y_pred = np.load("y_pred.npy")
print(y_true.shape, y_pred.shape)
cm = confusion_matrix(y_true, y_pred, normalize="true")

cm = np.around(cm*100, decimals=2)
plot_confusion_matrix(cm, ["Negative", "Neutral", "Positive"], title=None)
