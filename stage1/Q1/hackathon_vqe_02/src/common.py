#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/07/19 

import os
from pathlib import Path
from typing import Callable, Any, Union, List, Tuple, Dict

import numpy as np
from numpy import ndarray
from scipy.optimize import minimize
from openfermion.chem import MolecularData
from mindquantum.core.gates import H, X, Y, Z, RX, RY, RZ
from mindquantum.core.circuit import Circuit, UN
from mindquantum.simulator import Simulator
from mindquantum.core.operators import InteractionOperator, FermionOperator, normal_ordered
from mindquantum.core.operators import Hamiltonian, QubitOperator, TimeEvolution
from mindquantum.algorithm.nisq.chem import Transform
from mindquantum.algorithm.nisq.chem import (
  UCCAnsatz,
  QubitUCCAnsatz,
  HardwareEfficientAnsatz,
  generate_uccsd,
  uccsd_singlet_generator,
  uccsd_singlet_get_packed_amplitudes,
)
from qupack.vqe import ESConservation, ESConserveHam, ExpmPQRSFermionGate
ESConservation._check_qubits_max = lambda *args, **kwargs: None   # monkey-patching avoid FileNotFoundError

BASE_PATH = Path(__file__).parent
CACHE_PATH = BASE_PATH / '.cache' ; CACHE_PATH.mkdir(exist_ok=True)

PEEK = os.environ.get('peek', False)
DEBUG_HAM = os.environ.get('debug_ham', False)
DEBUG_ANSATZ = os.environ.get('debug_ansatz', False)

Ham = Union[Hamiltonian, ESConserveHam]
QVM = Union[Simulator, ESConservation]
Config = Dict[str, Any]
Params = ndarray


def get_ham(mol:MolecularData, is_fermi:bool=False) -> Ham:
  if is_fermi:
    ham_of = mol.get_molecular_hamiltonian()
    inter_ops = InteractionOperator(*ham_of.n_body_tensors.values())
    ham_hiq = FermionOperator(inter_ops)
    ham_fo = normal_ordered(ham_hiq)
    ham_op = ham_fo.real
    ham = ESConserveHam(ham_op)   # ham of a FermionOperator
  else:
    ham_of = mol.get_molecular_hamiltonian()
    inter_ops = InteractionOperator(*ham_of.n_body_tensors.values())
    ham_hiq = FermionOperator(inter_ops)
    ham_qo = Transform(ham_hiq).jordan_wigner()   # fermi => pauli
    ham_op = ham_qo.real      # FIXME: why discard imag part?
    ham = Hamiltonian(ham_op)     # ham of a QubitOperator

  if DEBUG_HAM:
    mat = ham.hamiltonian.matrix().todense()
    diag = np.diag(mat)
    breakpoint()

  return ham


def get_ansatz(mol:MolecularData, ansatz:str, config:Config, no_hfw:bool=False) -> Tuple[Circuit, List[float]]:
  # Construct hartreefock wave function circuit: |0...> -> |1...>
  # NOTE: this flip is important though do not know why; H does not work
  if not no_hfw:
    hartreefock_wfn_circuit = UN(X, mol.n_electrons)

  init_amp = None
  if ansatz == 'UCC':
    ansatz_circuit = UCCAnsatz(mol.n_qubits, mol.n_electrons, trotter_step=config['trotter']).circuit
  elif ansatz == 'QUCC':
    ansatz_circuit = QubitUCCAnsatz(mol.n_qubits, mol.n_electrons, trotter_step=config['trotter']).circuit
  elif ansatz == 'UCCSD':
    if not 'sugar':
      ansatz_circuit, init_amp, _, _, _, _ = generate_uccsd(mol, threshold=config['thresh'])
    else:
      ucc_fermion_ops = uccsd_singlet_generator(mol.n_qubits, mol.n_electrons, anti_hermitian=True)
      ucc_qubit_ops = Transform(ucc_fermion_ops).jordan_wigner()
      ansatz_circuit = TimeEvolution(ucc_qubit_ops.imag).circuit    # ucc_qubit_ops 中已经包含了复数因子 i 
      init_amp_ccsd = uccsd_singlet_get_packed_amplitudes(mol.ccsd_single_amps, mol.ccsd_double_amps, mol.n_qubits, mol.n_electrons)
      init_amp = [init_amp_ccsd[i] for i in ansatz_circuit.params_name]
  elif ansatz == 'UCCSD-QP':
    ucc_fermion_ops = uccsd_singlet_generator(mol.n_qubits, mol.n_electrons, anti_hermitian=False)
    circ = Circuit()
    for term in ucc_fermion_ops: circ += ExpmPQRSFermionGate(term)
  elif ansatz == 'HEA':
    rot_gates = [globals()[g] for g in config['rot_gates']]
    entgl_gate = globals()[config['entgl_gate']]
    ansatz_circuit = HardwareEfficientAnsatz(mol.n_qubits, rot_gates, entgl_gate, depth=config['depth']).circuit
  else:
    raise ValueError(f'unknown ansatz: {ansatz}')

  if ansatz == 'UCCSD-QP':
    vqc = circ
  elif no_hfw:
    vqc = ansatz_circuit
  else:
    vqc = hartreefock_wfn_circuit + ansatz_circuit

  if DEBUG_ANSATZ:
    vqc.summary()

  return vqc, init_amp


def run_expectaion(sim:QVM, ham:Ham, circ:Circuit, params:ndarray) -> float:
  # Construct parameter resolver of the taregt state circuit
  pr = dict(zip(circ.params_name, params))
  # Calculate energy of ground state
  if isinstance(sim, ESConservation):
    E = sim.get_expectation(ham, circ, pr)
  else:
    # Evolve into tagert state
    sim.apply_circuit(circ, pr)
    if 'sugar':
      E = sim.get_expectation(ham)
    else:
      qs1 = sim.get_qs()
      H = ham.hamiltonian.matrix().todense()
      E = (qs1 @ H @ qs1)

  return E.real
