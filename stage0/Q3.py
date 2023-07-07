#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/07/07 

# 问题描述：请将如上3比特线路进行比特序号的翻转，将之前作用在第0个特比上的门作用在第2个比特上，作用在第1个比特上的门保持不变。

import numpy as np
from mindquantum.utils import random_circuit

circ = random_circuit(3, 20, seed=42)

# 请补充如下代码片段
new_circ = circ.reverse_qubits()

print(new_circ)
