#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/07/15

from argparse import ArgumentParser
from solver import read_csv, get_mol

# due to *.csv read/write precision error for judger of this contest
# we do NOT use direct output in the program as the final FCI value
# we parse the csv file and reconstruct to get it :(

if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-i', '--input-mol',  help='input molecular *.csv',  default='h4.csv')
  args = parser.parse_args()

  name, geo = read_csv(args.input_mol)
  mol = get_mol(name, geo, run_fci=True)
  print('final fci energe (theoretical):', mol.fci_energy)
