#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/07/07 

# 问题描述：制备一个量子线路，量子线路中只有一个 `X` 门，该门被第0个和第1个比特控制，作用在第2个比特上。

from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import X

circ = Circuit([
  # 请补充如下代码片段
  X.on(obj_qubits=2, ctrl_qubits=[0, 1])
])

print(circ)
