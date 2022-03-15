import utils
import math
import numpy as np
from sklearn.model_selection import train_test_split
from parzen_window import parzenWindow
from k_nearest_neighbor import kNN


if __name__ == '__main__':
    size_sample = 1200  # 样本数
    size_kn = 10  # 最近邻的点数
    size_window = 10 # 窗宽
    dataset = []  # 原始数据集

    # 从数据文件中加载原始数据集
    with open("./data/samples_c3_s1500.txt", "r") as f:
        for line in list(f):
            dataset.append(eval(line)) if len(line) > 0 else None
        dataset = np.asarray(dataset, dtype=object)

    # 分割原始数据集得到指定大小的训练集和测试集
    dataset_spilt, _ = train_test_split(dataset, test_size=1 - size_sample / dataset.shape[0], random_state=0)
    dataset_train, dataset_test = train_test_split(dataset_spilt, test_size=0.2, random_state=0)

    # 转换数据集格式
    dataset_train, _ = utils.transferDataset(dataset_train, mode="train")
    dataset_test, label_test = utils.transferDataset(dataset_test, mode="test")

    # 通过kNN方法估计概率密度并分类
    knn_pdf_pred, knn_label_pred = kNN(dataset_test, dataset_train, kn_size=size_kn)

    # 通过ParzenWindow方法估计概率密度并分类
    pn_pdf_pred, pn_label_pred = parzenWindow(dataset_test, dataset_train, window_size=size_window)

    # 计算准确率
    acc = utils.calculateAcc(pred=knn_label_pred, gt=label_test)
    print("[kNN] size_sample:", size_sample)
    print("[kNN] size_kn:", size_kn)
    print("[kNN] Acc:", acc)
    acc = utils.calculateAcc(pred=pn_label_pred, gt=label_test)
    print("[Parzen Window] size_sample:", size_sample)
    print("[Parzen Window] size_window:", size_window)
    print("[Parzen Window] Acc:", acc)

    # 绘制结果-概率密度图
    utils.showPDF(xy=dataset_test, pdf=knn_pdf_pred)
    utils.showPDF(xy=dataset_test, pdf=pn_pdf_pred)

    # 绘制结果-分类估计图
    utils.showData(data=dataset, colormap=['#CD5555', '#104E8B', '#008B00'])

    temp = [[dataset_test[i], knn_label_pred[i]] for i in range(len(dataset_test))]
    utils.showData(data=temp, colormap=['#CD5555', '#104E8B', '#008B00'])
    temp = [[dataset_test[i], pn_label_pred[i]] for i in range(len(dataset_test))]
    utils.showData(data=temp, colormap=['#CD5555', '#104E8B', '#008B00'])
