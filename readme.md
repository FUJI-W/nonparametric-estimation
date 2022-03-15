### 说明

本项目基于Python语言，实现了**非参数概率密度估计**中的两种方法：①<u>**k近邻（k-nearest neighbor）**</u>方法；②<u>**Parzen窗（Parzen Window）**</u>方法。

------

### 文件树

```shell
└─nonparametric_estimation
    │  k_nearest_neighbor.py  # k近邻方法实现
    │  parzen_window.py  # Parzen窗方法实现
    │  utils.py  # 数据生成、处理与可视化的方法实现
    |  main.py  # 主函数
    │
    └─data
          |  samples.txt  # 测试数据集样例
```

------

### 运行

#### 生成数据集

- **命令行**

  ```
  python utils.py
  ```

- **参数**

  ```python
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
  ```

- **结果样例**

  |                           数据参数                           |                          样本点分布                          |                         概率密度分布                         |
  | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
  | ![image-20220315193614558](https://gitee.com/FujiW/pic-bed/raw/master/20220315193614.png) | ![image-20220315193345172](https://gitee.com/FujiW/pic-bed/raw/master/20220315193345.png) | ![image-20220315193243984](https://gitee.com/FujiW/pic-bed/raw/master/20220315193432.png) |

#### 仅运行 kNN 方法

- **命令行**

  ```
  python k_nearest_neighbor.py
  ```

- **参数**

  ```python
  size_sample = 1200  # 样本数
  size_kn = 10  # 最近邻的点数
  ```

- **结果样例**

  |                           运行参数                           |                           分类预测                           |                         概率密度估计                         |
  | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
  | ![image-20220315194403442](https://gitee.com/FujiW/pic-bed/raw/master/20220315194403.png) | ![image-20220315194234309](https://gitee.com/FujiW/pic-bed/raw/master/20220315194234.png) | ![image-20220315194223689](https://gitee.com/FujiW/pic-bed/raw/master/20220315194223.png) |

#### 仅运行 Parzen Window 方法

- **命令行**

  ```shell
  python parzen_window.py
  ```

- **参数**

  ```python
  size_sample = 1200  # 样本数
  size_window = 10  # 窗宽
  ```

- **结果样例**

  |                           运行参数                           |                           分类预测                           |                         概率密度估计                         |
  | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
  | ![image-20220315194850082](https://gitee.com/FujiW/pic-bed/raw/master/20220315194850.png) | ![image-20220315194955855](https://gitee.com/FujiW/pic-bed/raw/master/20220315194955.png) | ![image-20220315194917237](https://gitee.com/FujiW/pic-bed/raw/master/20220315194926.png) |

#### 同时运行两种方法

- **命令行**

  ```
  python main.py
  ```

- **参数**

  ```python
  size_sample = 1200  # 样本数
  size_kn = 10  # 最近邻的点数
  size_window = 10 # 窗宽
  ```

- **结果样例**

  |                         kNN 分类预测                         |                    Parzen Window 分类预测                    |                          数据集分布                          |
  | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
  | ![image-20220315195615111](https://gitee.com/FujiW/pic-bed/raw/master/20220315195743.png) | ![image-20220315195620918](https://gitee.com/FujiW/pic-bed/raw/master/20220315195750.png) | ![image-20220315201005048](https://gitee.com/FujiW/pic-bed/raw/master/20220315201005.png) |

  |                       kNN 概率密度估计                       |                  Parzen Window 概率密度估计                  |                         GT 概率密度                          |
  | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
  | ![image-20220315200236408](https://gitee.com/FujiW/pic-bed/raw/master/20220315201100.png) | ![](https://gitee.com/FujiW/pic-bed/raw/master/20220315201105.png) | ![image-20220315200849144](https://gitee.com/FujiW/pic-bed/raw/master/20220315201111.png) |

  