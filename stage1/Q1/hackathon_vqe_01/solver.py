import os ; os.environ['OMP_NUM_THREADS'] = '4'
import json
from time import time
from argparse import ArgumentParser
from typing import List, Tuple

import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mindquantum.core.operators import FermionOperator, InteractionOperator, normal_ordered
from mindquantum.core.circuit import Circuit
from mindquantum.algorithm.nisq.chem import uccsd_singlet_generator
from qupack.vqe import ESConservation, ESConserveHam, ExpmPQRSFermionGate
from openfermion.chem import MolecularData

# this experimental solver optimizes either of the 
#   - pyscf-computed real FCI
#   - VQE-approximated E0 energy prediction (QuPack)

OPTIM_METH = [
#  'Nelder-Mead',
#  'Powell',
  'CG',
#  'Newton-CG',
  'BFGS',
#  'L-BFGS-B',
#  'TNC',
  'COBYLA',
#  'SLSQP',
#  'dogleg',
  'trust-constr',
#  'trust-ncg',
#  'trust-exact',
#  'trust-krylov',
]
INIT_METH = [
  'randu', 
  'linear', 
#  'randn', 
#  'orig',
]

if 'typing':
  DTYPE = np.float64
  Name = List[str]
  Geo = ndarray     # [N*3], flattened

if 'globals':
  steps: int = 0
  circ: Circuit = None
  sim: ESConservation = None
  track_ene: List[float] = []
  track_geo: List[Geo]   = []

  # contest time limit: 1h
  TIMEOUT_LIMIT = int(3600 * 0.95)
  global_T = time()


def timer(fn):
  def wrapper(*args, **kwargs):
    from time import time
    start = time()
    r = fn(*args, **kwargs)
    end = time()
    print(f'[Timer]: {fn.__name__} took {end - start:.3f}s')
    return r
  return wrapper

def read_csv(fp:str) -> Tuple[Name, Geo]:
  with open(fp, 'r', encoding='utf-8') as fh:
    data = fh.readlines()

  mol_name, mol_geo = [], []
  for line in data:
    name, *geo = line.split(',')
    mol_name.append(name)
    mol_geo.extend([DTYPE(e) for e in geo])
  return mol_name, np.ascontiguousarray(mol_geo, dtype=DTYPE)

def write_csv(fp:str, name:Name, geo:Geo):
  items = [[n] + geo[i].tolist() for i, n in enumerate(name)]

  with open(fp, 'w', encoding='utf-8') as fh:
    for item in items:
      fh.write(', '.join([str(e) for e in item]))
      fh.write('\n')


# ↓↓↓ openfermionpyscf._run_pyscf.py: run_pyscf() ↓↓↓

'''
  - calc hf_energy, prepare integrals 
  - calc fci_energy (optional)
'''

from pyscf import fci
from openfermionpyscf import PyscfMolecularData
from openfermionpyscf._run_pyscf import prepare_pyscf_molecule, compute_scf, compute_integrals

def run_pyscf_hijack(molecule:MolecularData, run_fci:bool=False) -> PyscfMolecularData:
  # Prepare pyscf molecule.
  pyscf_molecule = prepare_pyscf_molecule(molecule)
  molecule.n_orbitals = int(pyscf_molecule.nao_nr())
  molecule.n_qubits = 2 * molecule.n_orbitals
  molecule.nuclear_repulsion = float(pyscf_molecule.energy_nuc())

  # Run SCF.
  pyscf_scf = compute_scf(pyscf_molecule)
  pyscf_scf.verbose = 0
  pyscf_scf.run()
  molecule.hf_energy = float(pyscf_scf.e_tot)

  # Hold pyscf data in molecule. They are required to compute density
  # matrices and other quantities.
  molecule._pyscf_data = pyscf_data = {}
  pyscf_data['mol'] = pyscf_molecule
  pyscf_data['scf'] = pyscf_scf

  # Populate fields.
  molecule.canonical_orbitals = pyscf_scf.mo_coeff.astype(float)
  molecule.orbital_energies = pyscf_scf.mo_energy.astype(float)

  # Get integrals.
  one_body_integrals, two_body_integrals = compute_integrals(pyscf_molecule, pyscf_scf)
  molecule.one_body_integrals = one_body_integrals
  molecule.two_body_integrals = two_body_integrals
  molecule.overlap_integrals = pyscf_scf.get_ovlp()

  # Run FCI.
  if run_fci:
    pyscf_fci = fci.FCI(pyscf_molecule, pyscf_scf.mo_coeff)
    pyscf_fci.verbose = 0
    molecule.fci_energy = pyscf_fci.kernel()[0]
    pyscf_data['fci'] = pyscf_fci

  # Return updated molecule instance.
  pyscf_molecular_data = PyscfMolecularData.__new__(PyscfMolecularData)
  pyscf_molecular_data.__dict__.update(molecule.__dict__)
  return pyscf_molecular_data

# ↑↑↑ openfermionpyscf._run_pyscf.py: run_pyscf() ↑↑↑


# ↓↓↓ VQE stuff ↓↓↓

def get_mol(name:Name, geo:Geo, run_fci:bool=False) -> MolecularData:
  geometry = [[name[i], list(e)] for i, e in enumerate(geo.reshape(len(name), -1))]
  mol = MolecularData(geometry, 'sto3g', multiplicity=1, data_directory='/tmp', filename='mol.h5')
  # NOTE: make integral calculation info for `mol.get_molecular_hamiltonian()`
  return run_pyscf_hijack(mol, run_fci)

def gen_ham(mol:MolecularData) -> ESConserveHam:
  ham_of = mol.get_molecular_hamiltonian()
  inter_ops = InteractionOperator(*ham_of.n_body_tensors.values())
  ham_hiq = FermionOperator(inter_ops)  # len(ham_hiq) == 1861 for LiH; == 37 for H2
  ham_fo = normal_ordered(ham_hiq)
  ham = ESConserveHam(ham_fo.real)      # len(ham.terms) == 631 for LiH; == 15 for H2
  #mat = ham.ham.matrix().todense()      # [4096, 4096] for LiH, 12-qubits; [16, 16] for H2, 4-qubits
  #diag = np.diag(mat)
  return ham

def gen_uccsd(mol:MolecularData) -> Circuit:
  ucc_fermion_ops = uccsd_singlet_generator(mol.n_qubits, mol.n_electrons, anti_hermitian=False)
  circ = Circuit()
  for term in ucc_fermion_ops: circ += ExpmPQRSFermionGate(term)
  return circ

def run_uccsd(ham:ESConserveHam, circ:Circuit, sim:ESConservation) -> float:
  grad_ops = sim.get_expectation_with_grad(ham, circ)

  def fun(tht, grad_ops):
    f, g = grad_ops(tht)
    return f.real, g.real

  tht = np.random.uniform(size=len(circ.params_name)) * 0.01
  res = minimize(fun, tht, args=(grad_ops,), jac=True, method=args.optim)
  return res.fun

# ↑↑↑ VQE stuff ↑↑↑


def optim_fn(geo:Geo, name:Name, objective:str):
  global circ, sim, steps

  if objective == 'pyscf':
    mol = get_mol(name, geo, run_fci=True)
    res: float = mol.fci_energy
  elif objective == 'uccsd':
    mol = get_mol(name, geo)    # only need *.h5 file
    ham = gen_ham(mol)
    if circ is None:
      circ = gen_uccsd(mol) ; circ.summary()
      sim = ESConservation(mol.n_qubits, mol.n_electrons)
    res = run_uccsd(ham, circ, sim)
  else:
    raise ValueError(f'unknown objective: {objective}')

  time_elapse = time() - global_T

  if args.track:
    steps += 1
    if steps % 10 == 0:
      print(f'>> [Step {steps}] energy: {res}, total time elapse: {time_elapse:.3f} s')
    track_ene.append(res)
    track_geo.append(geo.reshape(len(name), -1))
  
  if time_elapse > TIMEOUT_LIMIT:
    best_geo = geo.reshape(len(name), -1)
    write_csv(args.output_mol, name, best_geo)
    exit(0)

  return res


@timer
def run(args, name:Name, geo:Geo) -> Tuple[float, Name, Geo]:
  if args.init == 'randu':
    geo = np.random.uniform(low=-1.0, high=1.0, size=len(geo)) * args.init_w
  elif args.init == 'randn':
    geo = np.random.normal(loc=0.0, scale=1.0, size=len(geo)) * args.init_w
  elif args.init == 'linear':
    n_mol = len(name)
    x = np.linspace(0.0, args.init_w, n_mol)
    geo = np.zeros([n_mol, 3])  # [N, D=3]
    for i in range(n_mol): geo[i, -1] = x[i]
    geo = geo.flatten()
  init_x = geo.tolist()

  s = time()
  res = minimize(
    optim_fn, 
    x0=geo, 
    args=(name, args.objective), 
    method=args.optim, 
    tol=args.eps, 
    options={'maxiter':args.maxiter, 'disp':True}
  )
  t = time()
  best_x = res.x    # flattened

  if args.track:
    mol = get_mol(name, best_x, run_fci=True)
    fci = mol.fci_energy
    print('final fci energe (theoretical):', fci)

    track_geo_np: ndarray = np.stack(track_geo, axis=0)
    with open(os.path.join(args.log_dp, 'stats.json'), 'w', encoding='utf-8') as fh:
      data = {
        'args': vars(args),
        'final_fci': fci,
        'ts': t - s,
        'init_x': init_x,
        'energy': track_ene,
        'name': name,
        'geometry': track_geo_np.tolist(),
      }
      json.dump(data, fh, indent=2, ensure_ascii=False)

    energy, geometry = track_ene, track_geo_np

    if 'plot energy':
      plt.clf()
      plt.subplot(211) ; plt.plot(energy)                 ; plt.title('Energy')
      plt.subplot(212) ; plt.plot(np.log(np.abs(energy))) ; plt.title('log(|Energy|)')
      plt.tight_layout()
      plt.savefig(os.path.join(args.log_dp, 'energy.png'))

    if 'plot geometry':
      T, N, D = geometry.shape    # (T=480, N=4, D=3)
      plt.clf()
      nrow = int(N**0.5)
      ncol = int(np.ceil(N / nrow))
      for i in range(nrow):
        for j in range(ncol):
          idx = i * ncol + j
          plt.subplot(nrow, ncol, idx + 1)
          plt.plot(geometry[:, idx, 0], 'r', label='x')
          plt.plot(geometry[:, idx, 1], 'g', label='y')
          plt.plot(geometry[:, idx, 2], 'b', label='z')
          plt.title(name[idx])
      plt.suptitle('Geometry')
      plt.tight_layout()
      plt.savefig(os.path.join(args.log_dp, 'geometry.png'))

  return name, best_x


@timer
def run_all(args):
  args.track = True
  name, geo = read_csv(args.input_mol)

  def setup_exp(args) -> bool:
    # setup log_dp
    expname = f'O={args.optim}_i={args.init}'
    expname += f'_w={args.init_w}' if args.init != 'orig' else ''
    args.log_dp = os.path.abspath(os.path.join(args.log_path, expname))
    if os.path.exists(args.log_dp): return True
    os.makedirs(args.log_dp, exist_ok=True)
  
    # reset globals
    global steps, circ, sim, track_ene, track_geo

    steps = 0
    circ = None
    sim = None
    track_ene = []
    track_geo = []

    return False

  for optim in OPTIM_METH:
    args.optim = optim
    for init in INIT_METH:
      args.init = init

      done = setup_exp(args)
      if done: continue

      print(f'>> run optim={optim}, init={init}')
      run(args, name, geo)


@timer
def run_compound(args):
  name, best_x = read_csv(args.input_mol)

  configs = [
    'COBYLA',         # this is fast
    'trust-constr',   # this is precise
  ]

  for i, optim in enumerate(configs):
    print(f'>> round {i}: optim use {optim}')
    args.optim = optim
    args.init = args.init if i == 0 else 'orig'

    name, best_x = run(args, name, best_x)
    best_geo = best_x.reshape(len(name), -1)
    write_csv(args.output_mol, name, best_geo)


if __name__ == '__main__':
  parser = ArgumentParser()
  # run
  parser.add_argument('-i', '--input-mol',  help='input molecular *.csv',  default='h4.csv')
  parser.add_argument('-x', '--output-mol', help='output molecular *.csv', default='h4_best.csv')
  parser.add_argument('-Z', '--objective',  help='optim target objective', default='uccsd', choices=['pyscf', 'uccsd'])
  parser.add_argument('-O', '--optim',      help='optim method',           default='BFGS', choices=OPTIM_METH)
  parser.add_argument('--init',     help='init method',    default='randu', choices=INIT_METH)
  parser.add_argument('--init_w',   help='init weight',    default=1,    type=float)
  parser.add_argument('--eps',      help='tol eps',        default=1e-6, type=float)
  parser.add_argument('--maxiter',  help='max optim iter', default=500,  type=int)
  # dev
  parser.add_argument('--run_all',  help='run all optim-init grid search', action='store_true')
  parser.add_argument('--log_path', help='track log out base path', default='log', type=str)
  args = parser.parse_args()

  if args.run_all:
    print(f'[dev mode] objective: {args.objective}')
    run_all(args)

  else:
    print('[submit mode]')
    run_compound(args)
