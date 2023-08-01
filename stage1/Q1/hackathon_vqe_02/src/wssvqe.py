#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/07/20

# NOTE: this does NOT work, though implementation is fine :(
# implementation of weighted SS-VQE in "Subspace-search variational quantum eigensolver for excited states"

from .common import *
from .ssvqe import get_ortho_circuits


def run(mol:MolecularData, ham:Ham, config:Config) -> Tuple[float, float]:
  # Declare the simulator
  if isinstance(ham, ESConserveHam):
    sim = ESConservation(mol.n_qubits, mol.n_electrons)
  else:
    sim = Simulator('mqvector', mol.n_qubits)
  
  # Construct encoding circ for preparing orthogonal init state |ψj>
  q0_enc, q1_enc = get_ortho_circuits()
  # Construct U ansatz circuit: U|ψj>
  ansatz, init_amp = get_ansatz(mol, config['ansatz'], config, no_hfw=True)
  # Full circuit
  q0_circ = q0_enc + ansatz
  q1_circ = q1_enc + ansatz
  # Get the expectation and gradient calculating function: <φj|U'HU|φj>
  q0_grad_ops = sim.get_expectation_with_grad(ham, q0_circ)
  q1_grad_ops = sim.get_expectation_with_grad(ham, q1_circ)

  # Define the objective function to be minimized
  def func(x:ndarray, q0_grad_ops:Callable, q1_grad_ops:Callable, w:float) -> Tuple[float, ndarray]:
    f0, g0 = q0_grad_ops(x)
    f1, g1 = q1_grad_ops(x)
    f0, f1, g0, g1 = [np.squeeze(x) for x in [f0, f1, g0, g1]]
    return np.real(f0 + w * f1), np.real(g0 + w * g1)
  
  # Initialize amplitudes
  init_amp = init_amp if init_amp is not None else np.random.random(len(ansatz.all_paras)) - 0.5
  
  # Get Optimized result: min. E0 = <ψ(λ0)|H|ψ(λ0)>
  res = minimize(
    func,
    init_amp,
    args=(q0_grad_ops, q1_grad_ops, config['w']),
    method=config['optim'],
    jac=True,
    tol=config['tol'],
    options={
      'maxiter': config['maxiter'], 
      'disp': config.get('debug', False),
    },
  )

  # Get the energy
  f0 = run_expectaion(sim, ham, q0_circ, res.x)
  f1 = run_expectaion(sim, ham, q1_circ, res.x)

  print('E0 energy:', f0)
  print('E1 energy:', f1)
  return f0, f1


def wssqve_solver(mol:MolecularData, config:Config) -> float:
  ansatz: str = config['ansatz']
  ham = get_ham(mol, ansatz.endswith('QP'))

  # Find the lowest two energies by min. U|φ0> + w*U|φ1>, sat. <φ0|φ1> = 0
  assert 0.0 < config['w'] < 1.0
  gs_ene, es_ene = run(mol, ham, config)

  return es_ene
