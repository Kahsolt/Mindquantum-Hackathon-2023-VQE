#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/07/07 

# 问题描述：如上所述

import numpy as np
from scipy.linalg import expm
from mindquantum.core.operators import TimeEvolution, QubitOperator

h = QubitOperator('X0 Y1') + QubitOperator('Z0 Y1')
m1 = expm(-1j*h.matrix().toarray())     # <= target
print('m1.shape:', m1.shape)
print('m1.dtype:', m1.dtype)
print(m1)

for n in range(1, 10):
  # 请补充如下代码片段
  circ = TimeEvolution(h, time=n).circuit
  if n == 1: print(circ)

  m2 = circ.matrix()
  if np.allclose(m1[0, 0], m2[0, 0], atol=1e-2):    # NOTE: 我觉得这个有问题
    print(f"Found: {n}")
    break
