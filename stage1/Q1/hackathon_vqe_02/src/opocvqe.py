#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/07/20 

# implementation of one-pass OC-VQE, combining OC-VQE and SS-VQE

from .common import *
import mindspore as ms
ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")
import mindspore.nn as nn
from mindspore.common.parameter import Parameter
from mindquantum.framework import MQAnsatzOnlyLayer
from mindquantum.simulator.utils import GradOpsWrapper
from mindquantum.core.circuit import add_prefix

# save current reg term for check
punish_f: float = None


# ↓↓↓ mindquantum/simulator/mqsim.py ↓↓↓

from mindquantum.simulator.mqsim import MQSim

MQSim.get_expectation_with_grad

def grad_ops_hijack(
  self: MQSim,
  hams: List[Hamiltonian],
  circ_left: Circuit,     # gs
  circ_right: Circuit,    # es
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
  param_names = L_ansatz_params_name + R_ansatz_params_name
  param_values = np.concatenate([inputs0, inputs1], axis=0)

  f_g = self.sim.get_expectation_with_grad_non_hermitian_multi_multi(
    [i.get_cpp_obj() for i in hams],
    [i.get_cpp_obj(hermitian=True) for i in hams],
    circ_left.get_cpp_obj(),
    circ_left.get_cpp_obj(hermitian=True),
    circ_right.get_cpp_obj(),
    circ_right.get_cpp_obj(hermitian=True),
    np.array([[]]),
    param_values,
    [],
    param_names,
    self.sim,
    1,
    1,
  )

  res = np.squeeze(np.array(f_g))
  f, g = res[0], res[1:]
  return f, g

# ↑↑↑ mindquantum/simulator/mqsim.py ↑↑↑


def optim_scipy(func:Callable, init_amp:np.ndarray, grad_ops:Tuple[Callable], config:Config) -> List[float]:
  n_round = len(config['optims'])
  for i, optim in enumerate(config['optims']):
    print(f'[{i+1}/{n_round}] {optim["optim"]}')
    res = minimize(
      func,
      init_amp,
      args=tuple([*grad_ops, optim['beta'], optim['w']]),
      method=optim['optim'],
      jac=True,
      tol=optim['tol'],
      options={
        'maxiter': optim['maxiter'], 
        'disp': optim.get('debug', False),
      },
    )
    init_amp = res.x
  return res.x


def optim_mindspore(pqc:MQAnsatzOnlyLayer, config:Config) -> ndarray:
  optim_cls = getattr(nn.optim, config['optim'])
  optimizer = optim_cls(pqc.trainable_params(), learning_rate=config['lr'])
  train_pqc = nn.TrainOneStepCell(pqc, optimizer)
  for _ in range(config['maxiter']): train_pqc()
  return np.asarray(pqc.weight.data, dtype=np.float32)


def run(mol:MolecularData, ham:Ham, config:Config) -> Tuple[float, float]:
  # Declare the simulator
  if isinstance(ham, ESConserveHam):
    sim = ESConservation(mol.n_qubits, mol.n_electrons)
    HAM_NULL = ESConserveHam(FermionOperator(''))
  else:
    sim = Simulator('mqvector', mol.n_qubits)
    HAM_NULL = Hamiltonian(QubitOperator(''))

  # Construct state ansatz circuit: |ψ(λj)>
  gs_circ, init_amp0 = get_ansatz(mol, config['ansatz'], config)
  es_circ, init_amp1 = get_ansatz(mol, config['ansatz'], config)
  gs_circ = add_prefix(gs_circ, 'gs') ; len_gs = len(gs_circ.all_paras)
  es_circ = add_prefix(es_circ, 'es') ; len_es = len(es_circ.all_paras)
  # Get the expectation and gradient calculating function: <ψ(λj)|H|ψ(λj)>
  gs_grad_ops = sim.get_expectation_with_grad(ham, gs_circ)
  es_grad_ops = sim.get_expectation_with_grad(ham, es_circ)
  ip_grad_ops = lambda *inputs: grad_ops_hijack(sim.backend, [HAM_NULL], gs_circ, es_circ, *inputs)
  
  # Define the objective function to be minimized
  def func(x:ndarray, gs_grad_ops:Callable, es_grad_ops:Callable, ip_grad_ops:Callable, beta:float, w:float) -> Tuple[float, ndarray]:
    global punish_f
    
    x0 = x[:len_gs]
    x1 = x[-len_es:]
    f0, g0 = gs_grad_ops(x0)
    f1, g1 = es_grad_ops(x1)
    f0, f1, g0, g1 = [np.squeeze(x) for x in [f0, f1, g0, g1]]
    f_sum = f0 + w * f1
    g_cat = np.concatenate([g0, w * g1], axis=0)

    f_, g_ = ip_grad_ops(x0, x1)
    punish_f = beta * np.abs(f_) ** 2
    punish_g = beta * (np.conj(g_) * f_ + g_ * np.conj(f_))

    if PEEK: print('gs:', f0.real, 'es:', f1.real, 'reg:', punish_f)
    return np.real(f_sum + punish_f), np.real(g_cat + punish_g)

  # Initialize amplitudes
  if init_amp0 is not None and init_amp1 is not None:
    init_amp = init_amp0 + init_amp1
  else:
    init_amp = np.random.random(len_gs + len_es) - 0.5

  # Get optimized result
  if config['opt'] == 'scipy':
    params = optim_scipy(func, init_amp, (gs_grad_ops, es_grad_ops, ip_grad_ops), config)
  
  elif config['opt'] == 'mindspore':
    def grad_ops(x):
      f, g = func(x, gs_grad_ops, es_grad_ops, ip_grad_ops, config['beta'], config['w'])
      f = np.expand_dims(np.expand_dims(f, 0), 0)
      g = g[None, None, :]
      return f, g
    
    grad_ops_wrapper = GradOpsWrapper(
      grad_ops, 
      [ham], 
      es_circ, 
      gs_circ, 
      gs_circ.all_encoder.keys() + es_circ.all_encoder.keys(), 
      gs_circ.all_ansatz .keys() + es_circ.all_ansatz .keys(), 
      None,
    )

    pqc = MQAnsatzOnlyLayer(grad_ops_wrapper)
    pqc.weight = Parameter(ms.Tensor(init_amp, pqc.weight.dtype))
    params = optim_mindspore(pqc, config)
  
  else: raise ValueError(f'unknown optim lib: {config["opt"]}')

  # Get the energyies
  sim.reset() ; f0 = run_expectaion(sim, ham, gs_circ, params[:len_gs])
  sim.reset() ; f1 = run_expectaion(sim, ham, es_circ, params[-len_es:])
  
  print('E0 energy:', f0)
  print('E1 energy:', f1)
  return f0, f1


def opocvqe_solver(mol:MolecularData, config:Config) -> float:
  ansatz: str = config['ansatz']
  ham = get_ham(mol, ansatz.endswith('QP'))

  # Find the lowest two energies by min. U|φ0> + w*U|φ1> + β*<φ0|U'U|φ1>
  if 'optims' in config:
    for optim in config['optims']:
      assert 0.0 < optim['w'] < 1.0
  else:
    assert 0.0 < config['w'] < 1.0
  gs_ene, es_ene = run(mol, ham, config)

  print('reg_term:', punish_f)

  return es_ene
