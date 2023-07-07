#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/07/07 

from Q6 import *

from mindquantum.algorithm.nisq import HardwareEfficientAnsatz  # 导入HardwareEfficientAnsatz
from mindquantum.core.gates import RY  # 导入量子门RY

ansatz = HardwareEfficientAnsatz(
  # 请补充如下ansatz代码
  n_qubits=4, 
  single_rot_gate_seq=[RY], 
  entangle_gate=X, 
  entangle_mapping='linear',
  depth=3,
).circuit  # 通过HardwareEfficientAnsatz搭建Ansatz
ansatz.summary()  # 总结Ansatz
ansatz.svg()

print(ansatz)
