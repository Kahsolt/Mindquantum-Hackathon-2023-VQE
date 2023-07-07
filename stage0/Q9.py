#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/07/07 

from Q8 import *

import mindspore as ms  # 导入mindspore库并简写为ms
from mindquantum.framework import MQLayer  # 导入MQLayer
from mindquantum.simulator import Simulator

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
ms.set_seed(1)  # 设置生成随机数的种子
sim = Simulator('mqvector', circuit.n_qubits)
grad_ops = sim.get_expectation_with_grad(
  # 请补充如下代码片段，完成梯度算子搭建
  hams,
  circ_right=circuit,
)
QuantumNet = MQLayer(grad_ops)  # 搭建量子神经网络
print(QuantumNet)
