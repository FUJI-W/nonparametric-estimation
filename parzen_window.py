import math
import numpy as np
from sklearn.model_selection import train_test_split
import utils


def distFunc(a, b):
    """
    两个输入参数之间的距离函数
    :param a: 输入参数
    :param b: 输入参数
    :return: 两者的距离
    """
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def windowFunc(u, mode="gauss"):
    """
    窗函数
    :param u: 样本点与点x之间的距离
    :param mode: gauss-高斯窗/正态窗；cube-方窗；
    :return: 样本点对点x的概率密度估计的贡献值
    """
    assert mode == "gauss" or mode == "cube"
    if mode == "gauss":
        return 1.0 / (math.sqrt(2 * math.pi)) * math.e ** (-0.5 * u ** 2)
    elif mode == "cube":
        return 1.0 if u < 1 / 2 else 0.0


def parzenWindow(data_test, data_train, window_size=10):
    """
    ParzenWindow非参估计方法入口函数
    :param data_test: 测试数据
    :param data_train: 训练数据
    :param window_size: 调节窗宽大小
    :return: data_test中各点的概率密度值；预测的类别
    """
    pdf_pred = []  # 概率密度值
    label_pred = []  # 预测的类别

    # 遍历测试数据中的点x
    # 根据训练数据，计算不同类下点x处的概率密度，依此预测其类别
    for x in data_test:
        pdf, label = [], []
        for c in data_train.keys():
            n = len(data_train[c])  # 样本数
            h = window_size * 1.0 / math.sqrt(n)  # 窗宽
            V = h ** 2  # 窗口体积
            k = 0.0  # 落在窗口内的样本数
            for xi in data_train[c]:
                ki = windowFunc(u=distFunc(x, xi) / h, mode="gauss")
                k += ki
            p = k / n / V  # 类别c下点x处的概率密度
            pdf.append(p), label.append(c)
        pdf_pred.append(max(pdf))
        label_pred.append(label[pdf.index(max(pdf))])

    return pdf_pred, label_pred


if __name__ == '__main__':
    size_sample = 1200  # 样本数
    size_window = 10  # 窗宽
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

    # 通过ParzenWindow方法估计概率密度并分类
    pn_pdf_pred, pn_label_pred = parzenWindow(dataset_test, dataset_train, window_size=size_window)

    # 计算准确率
    acc = utils.calculateAcc(pred=pn_label_pred, gt=label_test)
    print("[Parzen Window] size_sample:", size_sample)
    print("[Parzen Window] size_window:", size_window)
    print("[Parzen Window] Acc:", acc)

    # 绘制结果-概率密度图
    utils.showPDF(xy=dataset_test, pdf=pn_pdf_pred)

    # 绘制结果-分类估计图
    utils.showData(data=dataset, colormap=['#CD5555', '#104E8B', '#008B00'])
    temp = [[dataset_test[i], pn_label_pred[i]] for i in range(len(dataset_test))]
    utils.showData(data=temp, colormap=['#CD5555', '#104E8B', '#008B00'])
    # showData(data=temp, colormap=['#FFC1C1', '#BBFFFF', '#9AFF9A'])
