#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/07/19 

# NOTE: this does NOT work, non-completed implementation :(
# implementation of SS-VQE in "Subspace-search variational quantum eigensolver for excited states"

from .common import *


def get_ortho_circuits() -> Tuple[Circuit, Circuit]:
  q0 = Circuit()
  q0 += X.on(0)
  q1 = Circuit()
  q1 += X.on(1)
  return q0, q1


def run_U(mol:MolecularData, ham:Ham, config:Config) -> Tuple[QVM, Tuple[Circuit, Circuit, Circuit]]:
  # Declare the U simulator
  if isinstance(ham, ESConserveHam):
    U_sim = ESConservation(mol.n_qubits, mol.n_electrons)
  else:
    U_sim = Simulator('mqvector', mol.n_qubits)
  # Construct encoding circ for preparing orthogonal init state
  q0_circ, q1_circ = get_ortho_circuits()
  # Construct U ansatz circuit: |ψ(λ0)>
  U_circ, init_amp = get_ansatz(mol, config['ansatz'], no_hfw=True)
  # Get the expectation and gradient calculating function: <φ|U'HU|φ>
  q0_grad_ops = U_sim.get_expectation_with_grad(ham, q0_circ + U_circ)
  q1_grad_ops = U_sim.get_expectation_with_grad(ham, q1_circ + U_circ)

  # Define the objective function to be minimized
  def func(x, q0_grad_ops, q1_grad_ops) -> Tuple[float, ndarray]:
    f0, g0 = q0_grad_ops(x)
    f1, g1 = q1_grad_ops(x)
    f0 = np.squeeze(f0)     # [1, 1] => []
    f1 = np.squeeze(f1)     # [1, 1] => []
    g0 = np.squeeze(g0)     # [1, 1, 96] => [96]
    g1 = np.squeeze(g1)     # [1, 1, 96] => [96]
    return np.real(f0 + f1), np.real(g0 + g1)
  # Initialize amplitudes
  init_amp = init_amp if init_amp is not None else np.random.random(len(U_circ.all_paras))
  # Get Optimized result: min. E0 = <ψ(λ0)|H|ψ(λ0)>
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

  f0, _ = q0_grad_ops(res.x)
  f1, _ = q1_grad_ops(res.x)

  print('E0 energy:', f0)
  print('E1 energy:', f1)
  return U_sim, (U_circ, q0_circ, q1_circ)


def run_V(mol:MolecularData, ham:Ham, config:Config, U_sim:QVM, U_circ:Circuit, q0_circ:Circuit, q1_circ:Circuit) -> float:
  # Declare the V simulator
  if isinstance(ham, ESConserveHam):
    V_sim = ESConservation(mol.n_qubits, mol.n_electrons)
  else:
    V_sim = Simulator('mqvector', mol.n_qubits)
  # Construct V ansatz circuit: VU|φj>
  V_circ, init_amp = get_ansatz(mol, config['ansatz'])
  # Get the expectation and gradient calculating function: <φj|U'V'HVU|φj>
  q0_grad_ops = V_sim.get_expectation_with_grad(ham, q0_circ + U_circ + V_circ)
  q1_grad_ops = V_sim.get_expectation_with_grad(ham, q1_circ + U_circ + V_circ)

  # Define the objective function to be maximized
  def gs_func(x, q0_grad_ops, q1_grad_ops) -> Tuple[float, ndarray]:
    f0, g0 = q0_grad_ops(x)
    f1, g1 = q1_grad_ops(x)
    f0 = np.squeeze(f0)     # [1, 1] => []
    f1 = np.squeeze(f1)     # [1, 1] => []
    g0 = np.squeeze(g0)     # [1, 1, 96] => [96]
    g1 = np.squeeze(g1)     # [1, 1, 96] => [96]
    return -np.real(f0 + f1), np.real(g0 + g1)
  # Initialize amplitudes
  init_amp = init_amp if init_amp is not None else np.random.random(len(U_circ.all_paras))
  # Get Optimized result: min. E0 = <ψ(λ0)|H|ψ(λ0)>
  res = minimize(
    gs_func,
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

  f0, _ = q0_grad_ops(res.x)
  f1, _ = q1_grad_ops(res.x)
  f0, f1 = f0.item(), f1.item()

  print('E0 energy:', min(f0, f1))
  print('E1 energy:', max(f0, f1))
  return max(f0, f1)


def ssqve_solver(mol:MolecularData, config1:Config, config2:Config) -> float:
  # we cannot freeze parameter of U when optimzing V
  # this is mainly a coding/programmatic problem :(
  raise NotImplemented

  ansatz1: str = config1['ansatz']
  ham = get_ham(mol, ansatz1.endswith('QP'), config)

  # Shrink the subspace, expand ansatz state: U|φ>
  gs_sim, *circuits = run_U(mol, ham, config1)
  # Find the highest energy Ek: VU|φ>
  es_ene = run_V(mol, ham, config2, gs_sim, *circuits)

  return es_ene
