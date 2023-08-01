#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/07/19 

# implementation of SS-VQE in "Subspace-search variational quantum eigensolver for excited states"
# NOTE: this does now work properly, still need investigation...

from .common import *
from mindquantum.core.circuit import add_prefix


def get_ortho_circuits() -> Tuple[Circuit, Circuit]:
  q0 = Circuit()
  q0 += X.on(0)
  q1 = Circuit()
  q1 += X.on(1)
  return q0, q1


def run_U(mol:MolecularData, ham:Ham, config:Config) -> QVM:
  # Declare the U simulator
  if isinstance(ham, ESConserveHam):
    sim = ESConservation(mol.n_qubits, mol.n_electrons)
  else:
    sim = Simulator('mqvector', mol.n_qubits)
  
  # Construct encoding circ for preparing orthogonal init state
  q0_enc, q1_enc = get_ortho_circuits()
  # Construct U ansatz circuit: |ψ(λ0)>
  U_circ, init_amp = get_ansatz(mol, config['ansatz'], config, no_hfw=True)
  # Full circuit
  q0_circ = q0_enc + U_circ ; q0_circ = add_prefix(q0_circ, 'U0')
  q1_circ = q1_enc + U_circ ; q1_circ = add_prefix(q1_circ, 'U1')
  # Get the expectation and gradient calculating function: <φ|U'HU|φ>
  q0_grad_ops = sim.get_expectation_with_grad(ham, q0_circ)
  q1_grad_ops = sim.get_expectation_with_grad(ham, q1_circ)

  # Define the objective function to be minimized
  def func(x:ndarray, q0_grad_ops:Callable, q1_grad_ops:Callable) -> Tuple[float, ndarray]:
    f0, g0 = q0_grad_ops(x)
    f1, g1 = q1_grad_ops(x)
    f0, f1, g0, g1 = [np.squeeze(x) for x in [f0, f1, g0, g1]]
    if PEEK: print('f0:', f0.real, 'f1:', f1.real)
    return np.real(f0 + f1), np.real(g0 + g1)
  
  # Initialize amplitudes
  if init_amp is None:
    init_amp = np.random.random(len(U_circ.all_paras)) - 0.5
  
  # Get optimized result
  res = minimize(
    func,
    init_amp,
    args=(q0_grad_ops, q1_grad_ops),
    method=config['optim'],
    jac=True,
    tol=config['tol'],
    options={
      'maxiter': config['maxiter'], 
      'disp': config.get('debug', False),
    },
  )

  # Get the energyies
  sim.reset() ; f0 = run_expectaion(sim, ham, q0_circ, res.x)
  sim.reset() ; f1 = run_expectaion(sim, ham, q1_circ, res.x)
  print('lowest energies:', [f0, f1])

  # arbitarily choose a circ to evolve into
  sim.reset()
  sim.apply_circuit(q0_circ, res.x)

  return sim


def run_V(mol:MolecularData, ham:Ham, config:Config, sim:QVM) -> float:
  # Construct V ansatz circuit: VU|φj>
  V_circ, init_amp = get_ansatz(mol, config['ansatz'], config)
  V_circ = add_prefix(V_circ, 'V')
  # Get the expectation and gradient calculating function: <φj|U'V'HVU|φj>
  grad_ops = sim.get_expectation_with_grad(ham, V_circ)

  # Define the objective function to be maximized
  def func(x:ndarray, grad_ops:Callable) -> Tuple[float, ndarray]:
    f, g = grad_ops(x)
    f, g = [np.squeeze(x) for x in [f, g]]
    if PEEK: print('f:', f.real)
    return -np.real(f), -np.real(g)
  
  # Initialize amplitudes
  if init_amp is None:
    init_amp = np.random.random(len(V_circ.all_paras)) - 0.5
  
  # Get Optimized result: min. E0 = <ψ(λ0)|H|ψ(λ0)>
  res = minimize(
    func,
    init_amp,
    args=(grad_ops),
    method=config['optim'],
    jac=True,
    tol=config['tol'],
    options={
      'maxiter': config['maxiter'], 
      'disp': config.get('debug', False),
    },
  )

  # NOTE: do not use `run_expectaion()` because it will reset QVM
  es_ene, _ = grad_ops(res.x)
  es_ene = es_ene.item().real

  print('E1 energy:', es_ene)
  return es_ene


def ssqve_solver(mol:MolecularData, config1:Config, config2:Config) -> float:
  ansatz1: str = config1['ansatz']
  ham = get_ham(mol, ansatz1.endswith('QP'))

  # Shrink the subspace, expand ansatz state: U|φ>
  sim = run_U(mol, ham, config1)
  # Find the highest energy Ek: VU|φ>
  es_ene = run_V(mol, ham, config2, sim)

  return es_ene
