import os
import sys
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(BASE_PATH))
sys.path.append(os.path.abspath(os.path.dirname(BASE_PATH)))
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(BASE_PATH))))

from openfermion.chem import MolecularData

from .ocvqe import ocqve_solver
from .opocvqe import opocvqe_solver
from .ssvqe import ssqve_solver
from .wssvqe import wssqve_solver

ANSATZS = [
  # mindquantum
  'QUCC',           # nice~
  'UCC',            # error very large ~0.1
  'UCCSD',          # error large ~0.01
  'HEA',            # FIXME: this does not work, really...?!!
  # qupack
  'UCCSD-QP',       # error very large ~0.1
]
OPTIMS = [
  'trust-constr',   # nice~
  'BFGS',           # error large ~0.01
  'CG',             # error large ~0.01
  'COBYLA',         # error very large ~1
]


def run_ocvqe(mol:MolecularData) -> float:
  config1 = {
    'ansatz':  'QUCC',
    'trotter': 2,
    'optim':   'BFGS',
    'tol':     1e-8,
    'dump':    False,
    'maxiter': 250,
    'debug':   False,
  }
  config2 = {
    'ansatz':  'QUCC',
    'trotter': 2,
    'optim':   'BFGS',
    'tol':     1e-8,
    'beta':    2,
    'eps':     2e-6,
    'maxiter': 300,
    'debug':   False,
    'cont_evolve': True,    # NOTE: trick
  }
  return ocqve_solver(mol, config1, config2)


def run_opocvqe(mol:MolecularData) -> float:
  config1 = {
    'ansatz':  'QUCC',
    'trotter': 1,
    'opt':     'scipy',
    'optims': [
      {
        'optim':   'CG',    # this is fast on converge 
        'tol':     1e-3,
        'beta':    10,
        'w':       0.2,
        'maxiter': 100,
      },
      {
        'optim':   'BFGS',  # this is accurate
        'tol':     1e-5,
        'beta':    100,
        'w':       0.1,
        'maxiter': 1000,
      },
    ]
  }
  config2 = {
    'ansatz':  'QUCC',
    'trotter': 1,
    'opt':     'mindspore',
    'optim':   'Adagrad',
    'lr':      0.15,
    'beta':    100,
    'w':       0.1,
    'maxiter': 1000,
  }
  return opocvqe_solver(mol, config1)


def run_ssvqe(mol:MolecularData) -> float:
  config1 = {
    'ansatz':     'HEA',
    'rot_gates':  ['RX', 'RY', 'RX'],
    'entgl_gate': 'X',
    'depth':      12,
    'optim':      'BFGS',
    'tol':        1e-8,
    'maxiter':    500,
    'debug':      False,
  }
  config2 = {
    'ansatz':     'HEA',
    'rot_gates':  ['RX', 'RY', 'RX'],
    'entgl_gate': 'X',
    'depth':      8,
    'optim':      'BFGS',
    'tol':        1e-8,
    'maxiter':    500,
    'debug':      False,
  }
  return ssqve_solver(mol, config1, config2)


def run_wssvqe(mol:MolecularData) -> float:
  config = {
    'ansatz':     'HEA',
    'rot_gates':  ['RX', 'RY', 'RX'],
    'entgl_gate': 'X',
    'depth':      12,
    'optim':      'BFGS',
    'tol':        1e-8,
    'w':          0.1,
    'maxiter':    1000,
    'debug':      False,
  }
  return wssqve_solver(mol, config)


def excited_state_solver(mol:MolecularData) -> float:
  #algo = 'ocvqe'
  algo = 'opocvqe'
  #algo = 'ssvqe'
  #algo = 'wssvqe'
  return globals()[f'run_{algo}'](mol)


class Main:
  def run(self, mol:MolecularData):
    return excited_state_solver(mol)
