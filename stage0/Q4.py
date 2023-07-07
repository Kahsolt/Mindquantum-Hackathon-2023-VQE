#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/07/07 

# 在变分量子算法中，我们通常会遇到很多结构一样，但是参数不一样的量子线路，在 MindQuantum 中，我们可以通过添加后缀的方式，改变参数化量子线路中的变量名。请完成如下代码，制备一个3层的纠缠线路。

from mindquantum.core.circuit import Circuit

circ = Circuit().rx('a', 1, 0).rx('b', 2, 1).rx('c', 0, 2)
print(circ)

# 请补充如下代码片段
from mindquantum.core.circuit import add_suffix
new_circ = Circuit()
for p in range(3):
  new_circ += add_suffix(circ, str(p))

print(new_circ)
