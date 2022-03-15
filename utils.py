import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import scipy.stats as st
from matplotlib.colors import ListedColormap


def generateDataset(category, mu, cov, size):
    """
    生成数据集（多元高斯分布）
    :param category: 类别
    :param mu: 多元高斯分布的均值
    :param cov: 多元高斯分布的方差
    :param size: 数据集的大小
    :return: 数据集
    """
    def normPDF(x, y, _mu, _cov):
        ret = st.multivariate_normal.pdf([x, y], _mu, _cov)
        return ret

    data = []
    for c in category:
        t_data = np.random.multivariate_normal(
            mean=mu[c],
            cov=cov[c],
            size=size[c]
        )
        data.extend([[[d[0], d[1]], int(c), normPDF(d[0], d[1], mu[c], cov[c])] for d in t_data])
    return data


def transferDataset(dataset_origin, mode="test"):
    """
    转换数据集格式
    :param dataset_origin: 原始数据集
    :param mode: 转换模式
    :return: 目标数据集（标签，仅对test数据集）
    """
    assert mode == "train" or mode == "test"
    # for train dataset: [[[x,y], category], ...] -> {'category': [[x,y], ...]}
    if mode == "train":
        dataset_target = {}
        for d in dataset_origin:
            if d[1] not in dataset_target.keys():
                dataset_target[d[1]] = []
            dataset_target[d[1]].append(d[0])
        return dataset_target, None
    # for test dataset:  [[[x,y], category], ...] -> [[x,y], ...] and [category, ...]
    else:
        dataset_target = np.asarray([[d[0][0], d[0][1]] for d in dataset_origin])
        label_target = [d[1] for d in dataset_origin]
        return dataset_target, label_target


def calculateAcc(pred, gt):
    """
    计算准确率
    :param pred: 预测数据列表
    :param gt: 真实数据列表
    :return: 准确率
    """
    return sum([int(pred[i] == gt[i]) for i in range(len(pred))]) / float(len(gt))


def showData(data, colormap):
    """
    绘制二维的数据分布图
    :param data: 二维数据
    :param colormap: 颜色映射表（颜色-类别）
    :return: 无返回值
    """
    pl.scatter(
        [d[0][0] for d in data],
        [d[0][1] for d in data],
        c=[d[1] for d in data],
        cmap=ListedColormap(colormap)
    )
    pl.show()


def showPDF(xy, pdf):
    """
    绘制三维的概率密度图
    :param xy: 二维点集
    :param pdf: 二维点对应的概率密度值
    :return: 无返回值
    """
    cor_X = xy[:, 0]
    cor_Y = xy[:, 1]
    cor_Z = pdf.copy()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_trisurf(cor_X, cor_Y, cor_Z, cmap='viridis', edgecolor='none')
    plt.show()


if __name__ == '__main__':
    is_save = False  # 是否保存到文件

    categories = ['0', '1', '2']  # 类别
    mus = {  # 均值
        categories[0]: [0, 0],
        categories[1]: [10, 10],
        categories[2]: [10, 0]
    }
    covs = {  # 协方差
        categories[0]: [[1, 0], [0, 10]],
        categories[1]: [[10, 0], [0, 1]],
        categories[2]: [[3, 0], [0, 4]]
    }
    sizes = {  # 样本数
        categories[0]: 500,
        categories[1]: 500,
        categories[2]: 500
    }

    # 生成数据集
    dataset = generateDataset(categories, mus, covs, sizes)

    # 绘制数据集概率密度图
    pdfs = [d[2] for d in dataset]
    xys = [[d[0][0], d[0][1]] for d in dataset]
    showPDF(np.asarray(xys), pdfs)

    # 绘制数据集二维分布图
    showData(dataset, colormap=['#CD5555', '#104E8B', '#008B00'])

    # 存储数据集
    if is_save:
        path_save = "./data/samples_c{}_s{}.txt".format(
            len(categories), sum(sizes[k] for k in sizes.keys()))
        with open(path_save, 'w') as f:
            for d in dataset:
                f.write(str(d)+"\n")
        print("Save dataset =>", path_save)
