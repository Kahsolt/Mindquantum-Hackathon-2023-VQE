#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/07/20 

# implementation of one-pass OC-VQE, combining OC-VQE and SS-VQE

from copy import deepcopy
from scipy.optimize import approx_fprime
from .common import *

# save current reg term for check
punish_f: float = None
punish_g_last: ndarray = None


# ↓↓↓ mindquantum/simulator/mqsim.py ↓↓↓

from mindquantum.simulator.mqsim import MQSim

# FIXME: this does NOT work :(
def grad_ops_hijack(
  self: MQSim,
  new_qvm: Callable[[None], MQSim],
  hams: List[Hamiltonian],
  circ_right: Circuit,
  circ_left: Circuit,
  *inputs: ndarray,
):
  # gs, es
  inputs0, inputs1 = inputs

  if not 'check <phi0|phi1>':
    qs0 = circ_left .get_qs(pr=inputs0)
    qs1 = circ_right.get_qs(pr=inputs1)
    ip = qs0 @ qs1
    print('ip:', ip)

  L_ansatz_params_name = circ_left .all_ansatz.keys()
  R_ansatz_params_name = circ_right.all_ansatz.keys()

  # freeze the right/es/inputs1 and calc the left/gs/inputs0
  tmp: MQSim = new_qvm() 
  tmp.apply_circuit(circ_right, inputs1)
  f0_g0 = self.sim.get_expectation_with_grad_non_hermitian_multi_multi(
    [i.get_cpp_obj() for i in hams],
    [i.get_cpp_obj(hermitian=True) for i in hams],
    # ↓↓↓ change order ↓↓↓
    circ_right.get_cpp_obj(),
    circ_right.get_cpp_obj(hermitian=True),
    circ_left.get_cpp_obj(),
    circ_left.get_cpp_obj(hermitian=True),
    # ↑↑↑ change order ↑↑↑
    np.array([[]]),
    inputs0,        # <= grad of this
    [],
    L_ansatz_params_name,
    tmp.sim,      # mindquantum._mq_vector.mqvector
    1,
    1,
  )

  # freeze the left/gs/inputs0 and calc the right/es/inputs1
  tmp: MQSim = new_qvm()
  tmp.apply_circuit(circ_left, inputs0)
  f1_g1 = self.sim.get_expectation_with_grad_non_hermitian_multi_multi(
    [i.get_cpp_obj() for i in hams],
    [i.get_cpp_obj(hermitian=True) for i in hams],
    circ_left.get_cpp_obj(),
    circ_left.get_cpp_obj(hermitian=True),
    circ_right.get_cpp_obj(),
    circ_right.get_cpp_obj(hermitian=True),
    np.array([[]]),
    inputs1,        # <= grad of this
    [],
    R_ansatz_params_name,
    tmp.sim,      # mindquantum._mq_vector.mqvector
    1,
    1,
  )

  res0 = np.squeeze(np.array(f0_g0))
  res1 = np.squeeze(np.array(f1_g1))
  f0, g0 = res0[0], res0[1:]
  f1, g1 = res1[0], res1[1:]
  f = (f0 + f1) / 2
  g = np.concatenate([g0, g1], axis=0)
  return f, g

# ↑↑↑ mindquantum/simulator/mqsim.py ↑↑↑


def run(mol:MolecularData, ham:Ham, config:Config) -> Tuple[float, float]:
  # Declare the simulator
  if isinstance(ham, ESConserveHam):
    sim = ESConservation(mol.n_qubits, mol.n_electrons)
    new_qvm = lambda: ESConservation(mol.n_qubits, mol.n_electrons)
    HAM_NULL = ESConserveHam(FermionOperator(''))
  else:
    sim = Simulator('mqvector', mol.n_qubits)
    new_qvm = lambda: Simulator('mqvector', mol.n_qubits).backend
    sim_ip = Simulator('mqvector', mol.n_qubits)
    HAM_NULL = Hamiltonian(QubitOperator(''))

  # Construct state ansatz circuit: |ψ(λj)>
  gs_circ, init_amp0 = get_ansatz(mol, config['ansatz'], config)
  es_circ, init_amp1 = get_ansatz(mol, config['ansatz'], config)
  len_gs = len(gs_circ.all_paras)
  len_es = len(es_circ.all_paras)
  # Get the expectation and gradient calculating function: <ψ(λj)|H|ψ(λj)>
  gs_grad_ops = sim.get_expectation_with_grad(ham, gs_circ)
  es_grad_ops = sim.get_expectation_with_grad(ham, es_circ)
  ip_grad_ops = lambda *inputs: grad_ops_hijack(sim_ip.backend, new_qvm, [HAM_NULL], es_circ, gs_circ, *inputs)

  def ip_func(x:ndarray) -> float:
    nonlocal sim, gs_circ, es_circ
    x0 = x[:len_gs]
    x1 = x[-len_es:]
    qs0 = gs_circ.get_qs(pr=x0)
    qs1 = es_circ.get_qs(pr=x1)
    return (qs0 @ qs1).real
  
  # Define the objective function to be minimized
  def func(x:ndarray, gs_grad_ops:Callable, es_grad_ops:Callable, ip_grad_ops:Callable, ip_func:Callable, beta:float, w:float, eps:float) -> Tuple[float, ndarray]:
    global punish_f, punish_g_last
    
    x0 = x[:len_gs]
    x1 = x[-len_es:]
    f0, g0 = gs_grad_ops(x0)
    f1, g1 = es_grad_ops(x1)
    f0, f1, g0, g1 = [np.squeeze(x) for x in [f0, f1, g0, g1]]
    f_sum = f0 + w * f1
    g_cat = np.concatenate([g0, w * g1], axis=0)

    # reg term
    sel = 'scipy'
    if sel == 'scipy':
      f_ = ip_func(x)
      g_ = approx_fprime(x, ip_func, epsilon=eps)
    elif sel == 'mq':
      f_, g_ = ip_grad_ops(x0, x1)

    punish_f = beta * np.abs(f_) ** 2
    if punish_f > eps:
      punish_g = beta * (np.conj(g_) * f_ + g_ * np.conj(f_))
      punish_g_last = punish_g
    else:
      punish_g = punish_g_last

    print('f0:', f0.real, 'f1:', f1.real, 'punish_f:', punish_f)
    return np.real(f_sum + punish_f), np.real(g_cat + punish_g)

  # Initialize amplitudes
  if init_amp0 is not None and init_amp1 is not None:
    init_amp = init_amp0 + init_amp1
  else:
    init_amp = np.random.random(len_gs + len_es) - 0.5

  # Get Optimized result
  res = minimize(
    func,
    init_amp,
    args=(gs_grad_ops, es_grad_ops, ip_grad_ops, ip_func, config['beta'], config['w'], config['eps']),
    method=config['optim'],
    jac=True,
    tol=config['tol'],
    options={
      'maxiter': config['maxiter'], 
      'disp': config.get('debug', False),
    },
  )

  # Get the energy
  f0 = run_expectaion(sim, ham, gs_circ, res.x[:len_gs])
  f1 = run_expectaion(sim, ham, es_circ, res.x[-len_es:])
  
  print('E0 energy:', f0)
  print('E1 energy:', f1)
  return f0, f1


def opocvqe_solver(mol:MolecularData, config:Config) -> float:
  ansatz: str = config['ansatz']
  ham = get_ham(mol, ansatz.endswith('QP'))

  # Find the lowest two energies by min. U|φ0> + w*U|φ1> + β*<φ0|U'U|φ1>
  assert 0.0 < config['w'] < 1.0
  gs_ene, es_ene = run(mol, ham, config)

  print('reg_term:', punish_f)

  return es_ene
