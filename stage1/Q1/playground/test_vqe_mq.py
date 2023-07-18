import os ; os.environ['OMP_NUM_THREADS'] = '2'
import json
from pathlib import Path
from time import time
from argparse import ArgumentParser
from typing import *

import numpy as np
from scipy.optimize import minimize
import mindspore as ms
from openfermion.chem import MolecularData
from mindspore import context
context.set_context(mode=context.PYNATIVE_MODE, device_target='CPU')
from mindspore.common.parameter import Parameter
from mindquantum.core.operators import InteractionOperator, FermionOperator, normal_ordered, TimeEvolution, Hamiltonian
from mindquantum.core.circuit import Circuit, UN
from mindquantum.core.gates import X
from mindquantum.framework import MQAnsatzOnlyLayer
from mindquantum.algorithm import generate_uccsd
from mindquantum.algorithm.nisq.chem import (
  Transform,
  get_qubit_hamiltonian,
  uccsd_singlet_generator,
  uccsd_singlet_get_packed_amplitudes,
)
from mindquantum.simulator import Simulator
from mindquantum.simulator.utils import GradOpsWrapper

try: from openfermionpyscf import run_pyscf
except: print('[warn] missing openfermionpyscf')
try: from qupack.vqe import ESConservation, ESConserveHam, ExpmPQRSFermionGate
except: print('[warn] missing qupack')

from mol_data import GEOMETRY, get_mol_geo

Optima = Tuple[float, float]

# VQE with UCCSD ansatz in mindquantum & qupack 
#   https://www.mindspore.cn/mindquantum/docs/zh-CN/r0.8/vqe_for_quantum_chemistry.html 
#   https://gitee.com/mindspore/mindquantum/blob/r0.8/tutorials/source/7.vqe_for_quantum_chemistry.py
# measure get_expectation_with_grad():
#   https://www.mindspore.cn/mindquantum/docs/zh-CN/r0.8/mindquantum.simulator.html?highlight=get_expectation#mindquantum.simulator.Simulator.get_expectation_with_grad

MQ_OPT = 'Adagrad'
MQ_LR = 4e-2


def write_json(args, energy:float, ts:float):
  exp_name = f'{args.molecule}_L={args.library}_O={args.optim}_I={args.init}_sugar={args.sugar}'

  if not args.log_fp.exists():
    init_data = {
      'energy': {},
      'ts': {},
    }
    with open(args.log_fp, 'w', encoding='utf-8') as fh:
      json.dump(init_data, fh, indent=2, ensure_ascii=False)

  with open(args.log_fp, 'r', encoding='utf-8') as fh:
    data = json.load(fh)
  
  if exp_name not in data['energy']: data['energy'][exp_name] = []
  data['energy'][exp_name].append(energy)
  if exp_name not in data['ts']: data['ts'][exp_name] = []
  data['ts'][exp_name].append(ts)

  with open(args.log_fp, 'w', encoding='utf-8') as fh:
    json.dump(data, fh, indent=2, ensure_ascii=False)


def get_mol() -> MolecularData:
  geometry = get_mol_geo(args.molecule, fmt='mq')
  basis = 'sto3g'
  spin = 0
  print('Geometry:', geometry)

  mol = MolecularData(geometry, basis, multiplicity=2*spin+1, data_directory='/tmp', filename='mol')
  try:
    s = time()
    mol = run_pyscf(mol, run_scf=True, run_mp2=True, run_cisd=True, run_ccsd=True, run_fci=True, verbose=False)
    t = time()
    print(f'[run_pyscf] cost: ({t - s:.5}s)')
    print(f'  HF   energy: {mol.hf_energy  :20.16f} Ha')    # Hartree-Fock
    print(f'  MP2  energy: {mol.mp2_energy :20.16f} Ha')
    print(f'  CCSD energy: {mol.ccsd_energy:20.16f} Ha')
    print(f'  CISD energy: {mol.cisd_energy:20.16f} Ha')
    print(f'  FCI  energy: {mol.fci_energy :20.16f} Ha')
  except: print('[warn] ignore run_pyscf() for ground truth')

  try:
    mol.save()
    print(mol.filename)
  except: pass
  return mol


def optim_scipy(func:Callable, init_amp:np.ndarray, grad_ops:Callable) -> Optima:
  s = time()
  res = minimize(func, init_amp, args=(grad_ops,), method='BFGS', jac=True, tol=1e-6)
  t = time()
  print(f'energy: {res.fun}, steps: {res.nfev} ({t - s:.5}s)')
  return res.fun, t - s


def optim_mindquantum(mol_pqc:MQAnsatzOnlyLayer, lr:float=MQ_LR, eps:float=1e-6) -> Optima:
  s = time()
  optimizer = getattr(ms.nn, MQ_OPT)(mol_pqc.trainable_params(), learning_rate=lr)
  train_pqc = ms.nn.TrainOneStepCell(mol_pqc, optimizer)

  initial_energy = mol_pqc().asnumpy()
  print(f"Initial energy: {float(initial_energy):20.16f}")

  energy_diff = eps * 1000
  energy_last = initial_energy + energy_diff
  iter_idx = 0
  while abs(energy_diff) > eps:
    energy_i = train_pqc().asnumpy()
    if iter_idx % 5 == 0:
      print(f"Step {int(iter_idx):3} energy {float(energy_i):20.16f}")
    energy_diff = energy_last - energy_i
    energy_last = energy_i
    iter_idx += 1
  t = time()

  print(f'energy: {energy_i.item()}, steps: {iter_idx - 1} ({t - s:.5}s)')
  return energy_i.item(), t - s


def optim_go(args, func:Callable, grad_ops:Callable, n_params:int, init_amp:Circuit=None) -> Optima:
  if   args.init == 'zeros':   init_amp = np.zeros              (n_params) + 1e-6
  elif args.init == 'uniform': init_amp = np.random.uniform(size=n_params) * 1e-6
  elif args.init == 'normal':  init_amp = np.random.normal (size=n_params) * 1e-6
  elif args.init == 'ccsd':    assert init_amp is not None

  if args.optim == 'sp':
    return optim_scipy(func, init_amp, grad_ops)
  elif args.optim == 'mq':
    mol_pqc = MQAnsatzOnlyLayer(grad_ops, args.init if args.init != 'ccsd' else 'zeros')
    if args.init == 'ccsd':   # override 'zeros'
      mol_pqc.weight = Parameter(ms.Tensor(init_amp, mol_pqc.weight.dtype))
    return optim_mindquantum(mol_pqc)


def run_mindquantum(args, mol:MolecularData) -> Optima:
  if args.sugar >= 2:
    ansatz_circuit, init_amp, ansatz_param_names, ham_qo, n_qubits, n_electrons = generate_uccsd(mol, threshold=-1)
    ham = Hamiltonian(ham_qo.real)
  else:
    if args.sugar >= 1:
      ham_qo = get_qubit_hamiltonian(mol)
    else:
      ham_of = mol.get_molecular_hamiltonian()                         # 获得分子对应的哈密顿量
      inter_ops = InteractionOperator(*ham_of.n_body_tensors.values()) # 转化为相互作用算符
      ham_hiq = FermionOperator(inter_ops)                             # 转化为费米算符
      ham_qo = Transform(ham_hiq).jordan_wigner()
    ham = Hamiltonian(ham_qo.real)

    ucc_fermion_ops = uccsd_singlet_generator(mol.n_qubits, mol.n_electrons, anti_hermitian=True)
    ucc_qubit_ops = Transform(ucc_fermion_ops).jordan_wigner()
    ansatz_circuit = TimeEvolution(ucc_qubit_ops.imag).circuit   # ucc_qubit_ops 中已经包含了复数因子 i 

  hartreefock_wfn_circuit = UN(X, mol.n_electrons)
  circ = hartreefock_wfn_circuit + ansatz_circuit
  circ.summary()

  sim = Simulator('mqvector', circ.n_qubits)
  grad_ops: GradOpsWrapper = sim.get_expectation_with_grad(ham, circ)

  def func(amp:np.ndarray, grad_ops:GradOpsWrapper) -> Tuple[float, np.ndarray]:
    f, g = grad_ops(amp)
    f = np.real(np.squeeze(f))
    g = np.real(np.squeeze(g))
    return f, g
  
  init_amp = None
  if args.init == 'ccsd':   # use estimated data from run_pyscf()
    init_amp_ccsd = uccsd_singlet_get_packed_amplitudes(mol.ccsd_single_amps, mol.ccsd_double_amps, mol.n_qubits, mol.n_electrons)
    init_amp = np.asarray([init_amp_ccsd[i] for i in ansatz_circuit.params_name])
  
  return optim_go(args, func, grad_ops, n_params=len(circ.params_name), init_amp=init_amp)


def run_qupack(args, mol:MolecularData) -> Optima:
  ham_of = mol.get_molecular_hamiltonian()                         # 获得分子对应的哈密顿量
  inter_ops = InteractionOperator(*ham_of.n_body_tensors.values()) # 转化为相互作用算符
  ham_hiq = FermionOperator(inter_ops)                             # 转化为费米算符
  ham_fo = normal_ordered(ham_hiq).real                            # 对费米算符的各项进行排序
  ham = ESConserveHam(ham_fo)                                      # 构造自旋数和电子数守恒的哈密顿量

  ucc_fermion_ops = uccsd_singlet_generator(mol.n_qubits, mol.n_electrons, anti_hermitian=False)
  circ = Circuit()
  for term in ucc_fermion_ops: circ += ExpmPQRSFermionGate(term)
  circ.summary()

  sim = ESConservation(mol.n_qubits, mol.n_electrons)
  grad_ops = sim.get_expectation_with_grad(ham, circ)

  def func(amp:np.ndarray, grad_ops) -> Tuple[float, np.ndarray]:
    f, g = grad_ops(amp)      # amp是线路参数值，f是期望值，g是参数梯度
    return f.real, g.real

  return optim_go(args, func, grad_ops, n_params=len(circ.params_name))


def run_all(mol:MolecularData):
  args.library = 'mq'
  for o in ['sp', 'mq']:
    args.optim = o
    for i in ['zeros', 'normal', 'uniform', 'ccsd']:
      args.init = i
      ene, ts = run_mindquantum(args, mol)
      write_json(args, ene, ts)

  args.library = 'qp'
  args.optim = 'sp'
  for i in ['zeros', 'normal', 'uniform']:
    args.init = i
    ene, ts = run_qupack(args, mol)
    write_json(args, ene, ts)


def run_all_all(mol:MolecularData):
  global MQ_LR, MQ_OPT

  for opt in ['Adagrad', 'Adam', 'SGD']:
    MQ_OPT = opt
    for lr in [1e-2, 4e-2, 6e-2, 1e-1]:
      MQ_LR = lr

      run_all(mol)


if __name__ == '__main__':
  LOG_FILE = Path(__file__).absolute().with_suffix('.json')

  parser = ArgumentParser()
  parser.add_argument('-M', '--molecule', default='LiH_1.5', choices=GEOMETRY.keys(), help='molecule name')
  parser.add_argument('-L', '--library',  default='qp',    choices=['qp', 'mq'], help='library qupack or mindquantum')
  parser.add_argument('-O', '--optim',    default='sp',    choices=['sp', 'mq'], help='optimzer scipy.optimize or mindquantum')
  parser.add_argument('-I', '--init',     default='zeros', choices=['zeros', 'normal', 'uniform', 'ccsd'], help='optimzer scipy.optimize or mindquantum')
  parser.add_argument('--sugar', default=0, type=int, help='prefer sugar API')
  parser.add_argument('--run_all', action='store_true', help='run all library-optim pair')
  parser.add_argument('--run_all_all', action='store_true', help='run all library-optim-lr pair')
  parser.add_argument('--log_fp', default=LOG_FILE, type=Path, help='record perf result')
  args = parser.parse_args()

  mol = get_mol()

  if args.run_all_all:
    run_all_all(mol)
    exit(0)
  if args.run_all:
    run_all(mol)
    exit(0)
  
  if args.library == 'mq':
    run_mindquantum(args, mol)
  elif args.library == 'qp':
    run_qupack(args, mol)
