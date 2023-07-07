#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/07/07 

if 'prepare data':
  import numpy as np
  from numpy import ndarray
  from sklearn.datasets import load_iris
  from sklearn.model_selection import train_test_split

  iris_dataset = load_iris()
  X: ndarray = iris_dataset.data[:100, :].astype(np.float32)  # 选取iris_dataset的data的前100个数据，将其数据类型转换为float32，并储存在X中
  X_feature_names = iris_dataset.feature_names                # 将iris_dataset的特征名称储存在X_feature_names中
  y: ndarray = iris_dataset.target[:100].astype(int)          # 选取iris_dataset的target的前100个数据，将其数据类型转换为int，并储存在y中
  y_target_names = iris_dataset.target_names[:2]              # 选取iris_dataset的target_names的前2个数据，并储存在y_target_names中

  print(X.shape)          # 打印样本的数据维度
  print(X_feature_names)  # 打印样本的特征名称
  print(y_target_names)   # 打印样本包含的亚属名称
  print(y)                # 打印样本的标签的数组
  print(y.shape)

  alpha = X[:, :3] * X[:, 1:]       # 每一个样本中，利用相邻两个特征值计算出一个参数，即每一个样本会多出3个参数（因为有4个特征值），并储存在alpha中
  X = np.append(X, alpha, axis=1)   # 在axis=1的维度上，将alpha的数据值添加到X的特征值中
  print(X.shape)          # 打印此时X的样本的数据维度

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)  # 将数据集划分为训练集和测试集

  print(X_train.shape)    # 打印训练集中样本的数据类型
  print(X_test.shape)


from mindquantum.core.circuit import Circuit  # 导入Circuit模块，用于搭建量子线路
from mindquantum.core.circuit import UN       # 导入UN模块
from mindquantum.core.gates import H, X, RZ   # 导入量子门H, X, RZ

encoder = Circuit()                 # 初始化量子线路
encoder += UN(H, 4)                 # H门作用在每1位量子比特
for i in range(4):
  encoder += RZ(f'alpha{i}').on(i)  # RZ(alpha_i)门作用在第i位量子比特
for j in range(3):
  # 请补充如下代码片段
  encoder += X.on(j+1, j)
  encoder += RZ(f'alpha{j+4}').on(j+1)
  encoder += X.on(j+1, j)

print(encoder)
encoder = encoder.no_grad()  # Encoder作为整个量子神经网络的第一层，不用对编码线路中的梯度求导数，因此加入no_grad()
encoder.summary()            # 总结Encoder
encoder.svg()
