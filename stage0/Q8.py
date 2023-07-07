#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/07/07 

from Q7 import *

circuit = encoder.as_encoder() + ansatz.as_ansatz()  # 完整的量子线路由Encoder和Ansatz组成
circuit.summary()
circuit.svg()

print(circuit)

from mindquantum.core.operators import QubitOperator  # 导入QubitOperator模块，用于构造泡利算符
from mindquantum.core.operators import Hamiltonian    # 导入Hamiltonian模块，用于构建哈密顿量

hams = [
  # 请补充如下代码片段，完成赛题要求。
  Hamiltonian(QubitOperator('Z2', 1)),
  Hamiltonian(QubitOperator('Z3', 1)),
]  # 分别对第2位和第3位量子比特执行泡利Z算符测量，且将系数都设为1，构建对应的哈密顿量
print(hams)
