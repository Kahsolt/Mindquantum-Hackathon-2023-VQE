#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/07/14

import os
import json
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import matplotlib
matplotlib.rcParams.update({'font.size': 8})
from matplotlib.axes import Axes
import matplotlib.pyplot as plt

OPTIM_METH = [
  # good
  'trust-constr',
  'BFGS',
  'CG',
  'COBYLA',
  # bad
  #'Nelder-Mead',
  #'SLSQP',
  # slow
  #'Powell',
]
INIT_METH = [
  # good
  'randu', 
  'linear', 
  # ok
  #'orig',
  # bad
  #'randn', 
]
INIT_W = 1


def run(args):
  len_x = len(OPTIM_METH)
  len_y = len(INIT_METH)
  fci = np.zeros([len_x, len_y])
  ts  = np.ones([len_x, len_y]) * -1

  init_w = INIT_W
  for i, optim in enumerate(OPTIM_METH):
    for j, init in enumerate(INIT_METH):
      expname = f'O={optim}_i={init}'
      expname += f'_w={init_w}' if init != 'orig' else ''
      fp = os.path.join(args.log_path, expname, 'stats.json')

      with open(fp, 'r', encoding='utf-8') as fh:
        data = json.load(fh)
      fci[i, j] = data['final_fci']
      ts [i, j] = data['ts']
  
  print('FCI:', fci.shape)
  print(fci)

  def draw_heatmap(ax:Axes, data:np.ndarray, is_time:bool=False):
    ax.imshow(data.T)
    ax.set_xticks(range(len_x), OPTIM_METH)
    ax.set_yticks(range(len_y), INIT_METH)
    for i in range(len(OPTIM_METH)):
      for j in range(len(INIT_METH)):
        if is_time:
          ax.text(i, j, f'{data[i, j]:.2f}', ha='center', va='center', color='w')
        else:
          ax.text(i, j, f'{data[i, j]:.7f}', ha='center', va='center', color='w')
  
  plt.clf()
  fig, [ax1, ax2] = plt.subplots(2, 1)
  draw_heatmap(ax1, fci)      ; ax1.set_title('FCI')
  draw_heatmap(ax2, ts, True) ; ax2.set_title('run time cost')
  plt.tight_layout()
  fp = os.path.join(args.log_path, 'fci_cmp.png')
  plt.savefig(fp, dpi=600)
  print(f'>> savefig to {fp}')


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('--log_path', default='log', type=Path, help='log file folder')
  args = parser.parse_args()

  assert args.log_path.is_dir()
  run(args)
