#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/07/13 

GEOMETRY = {
  'H2': [        # 一个 H-H 分子，键长 0.74A
    ('H', [0.0, 0.0, 0.0 ]),
    ('H', [0.0, 0.0, 0.74]),
  ],
  'LiH': [
    ('Li', [0.0, 0.0, 0.0]),
    ('H',  [0.0, 0.0, 1.5]),
  ],
}


def get_mol_geo(name:str, fmt='mq'):
  assert fmt in ['chq', 'mq']

  data = GEOMETRY[name]
  if fmt == 'mq': return data
  
  for i, v in enumerate(data):
    data[i] = ' '.join([str(e) for e in [v[0]] + v[1]])
  return data
