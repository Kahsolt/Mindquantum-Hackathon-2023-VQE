#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/07/18

# benchmark all available VQE ansatz on E0 estimation

import os
import sys
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_PATH))
sys.path.append(os.path.join(BASE_PATH, '..', 'hackathon_vqe_01'))
sys.path.append(os.path.join(BASE_PATH, '..', 'hackathon_vqe_02'))
OUT_PATH = os.path.join(BASE_PATH, 'benchmark_results')
os.makedirs(OUT_PATH, exist_ok=True)
from time import time
from typing import Tuple, List

import numpy as np
from matplotlib.axes import Axes
import matplotlib
matplotlib.rcParams.update({'font.size': 6})
import matplotlib.pyplot as plt

from mindquantum.core.operators import FermionOperator, InteractionOperator, normal_ordered
from mindquantum.core.operators import Hamiltonian, QubitOperator
from mindquantum.algorithm.nisq.chem import Transform
from qupack.vqe import ESConserveHam
from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf, PyscfMolecularData

from mol_data import GEOMETRY
from src.main import ANSATZS, run_gs
from src import main as mod

# run_gs config
config = {
  'ansatz':  None,
  'optim':   'trust-constr',
  'tol':     1e-5,
  'maxiter': 500,
  'debug':   False,
  'dump':    False,
}


def get_mol(atoms:List[Tuple[str, float, float, float]]) -> PyscfMolecularData:
  mol = MolecularData(atoms, 'sto3g', multiplicity=1, data_directory='/tmp', filename='mol')
  return run_pyscf(mol, run_ccsd=True, run_fci=True)

def get_hams(mol:MolecularData) -> Tuple[Hamiltonian, ESConserveHam]:
  ham_of = mol.get_molecular_hamiltonian()
  inter_ops = InteractionOperator(*ham_of.n_body_tensors.values())
  ham_hiq = FermionOperator(inter_ops)

  if 'pauli':
    ham_qo = Transform(ham_hiq).jordan_wigner()     # fermi => pauli
    ham_op = ham_qo.real              # FIXME: why discard imag part?
    ham_p = Hamiltonian(ham_op)       # ham of a QubitOperator

  if 'fermi':
    ham_fo = normal_ordered(ham_hiq)
    ham_qo = ham_fo.real
    ham_f = ESConserveHam(ham_qo)     # ham of a FermionOperator

  return ham_p, ham_f


def run():
  pANSATZS = ['FCI'] + ANSATZS
  n_mol    = len(GEOMETRY)
  n_ansatz = len(pANSATZS)
  ene = np.zeros([n_mol, n_ansatz])
  ts  = np.zeros([n_mol, n_ansatz])

  for i, atoms in enumerate(GEOMETRY.values()):
    mol = get_mol(atoms)
    ene[i, 0] = mol.fci_energy
    ham_p, ham_f = get_hams(mol)

    for j, ansatz in enumerate(ANSATZS, start=1):
      config['ansatz'] = ansatz
      ham = ham_f if ansatz.endswith('QP') else ham_p
      s = time()
      _, e = run_gs(mol, ham, config)
      t = time()

      ts [i, j] = t - s
      ene[i, j] = e

  def plot_heatmap(data:np.ndarray, title:str, save_name:str, is_time:bool=False):
    plt.clf()
    plt.figure(figsize=(10, 6))
    plt.imshow(data.T)
    ax: Axes = plt.gca()
    ax.set_xticks(range(n_mol), GEOMETRY.keys())
    ax.set_yticks(range(n_ansatz), pANSATZS)
    for i in range(len(GEOMETRY.keys())):
      for j in range(len(pANSATZS)):
        if is_time:
          ax.text(i, j, f'{data[i, j]:.2f}', ha='center', va='center', color='w')
        else:
          ax.text(i, j, f'{data[i, j]:.7f}', ha='center', va='center')
    plt.suptitle(f'{mod.OPTIM1} - {title}')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_PATH, f'benchmark-{save_name}_{mod.OPTIM1}.png'), dpi=600)
    np.savetxt(os.path.join(OUT_PATH, f'benchmark-{save_name}_{mod.OPTIM1}.txt'), data.T)

  err = np.abs(ene - np.expand_dims(ene[:, 0], axis=-1))
  plot_heatmap(ene, title='Energy',       save_name='energy')
  plot_heatmap(err, title='Energy error', save_name='error')
  plot_heatmap(ts,  title='Time cost',    save_name='timecost')


if __name__ == '__main__':
  run()
