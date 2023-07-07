#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/07/07 

# 问题描述：如上左侧线路只作用在第0个、第1个和第3个比特，请移除不必要的比特，对量子线路进行压缩，完成如上右侧线路。

from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import SWAP
circ = Circuit().h(0).x(0,3)
circ += SWAP([0, 1])

# 请补充如下代码片段
new_circ = circ.compress()

print(new_circ)
