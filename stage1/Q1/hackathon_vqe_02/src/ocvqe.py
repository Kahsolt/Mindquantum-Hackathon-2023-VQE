#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/07/19 

# implementation of OC-VQE in "Variational Quantum Computation of Excited States"

import json
from .common import *

# save current reg term for check
punish_f: float = None


def run_gs(mol:MolecularData, ham:Ham, config:Config) -> Tuple[QVM, float, Params]:
  # Declare the ground state simulator
  if isinstance(ham, ESConserveHam):
    gs_sim = ESConservation(mol.n_qubits, mol.n_electrons)
  else:
    gs_sim = Simulator('mqvector', mol.n_qubits)
  
  # Construct ground state ansatz circuit: |ψ(λ0)>
  gs_circ, init_amp = get_ansatz(mol, config['ansatz'], config)

  # load cached if exists
  if config['dump']:
    fp = CACHE_PATH / f'ocvqe-{config["ansatz"]}-{Path(mol.filename).name}.json'
    if fp.exists():
      print('>> try using cached gs_sim states')
      try:
        with open(fp, 'r', encoding='utf-8') as fh:
          gs_pr = json.load(fh)
        gs_ene = run_expectaion(gs_sim, gs_circ, gs_pr)
        print('E0 energy:', gs_ene)
        return gs_sim, gs_ene, np.array(list(gs_pr.values()))
      except:
        print('>> cache file error')
        fp.unlink()
  
  # Get the expectation and gradient calculating function: <ψ(λ0)|H|ψ(λ0)>
  gs_grad_ops = gs_sim.get_expectation_with_grad(ham, gs_circ)

  # Define the objective function to be minimized
  def gs_func(x, grad_ops) -> Tuple[float, ndarray]:
    f, g = grad_ops(x)
    f = np.squeeze(f)     # [1, 1] => []
    g = np.squeeze(g)     # [1, 1, 96] => [96]
    if PEEK: print('gs:', f.real)
    return np.real(f), np.real(g)
  
  # Initialize amplitudes
  if init_amp is None:
    init_amp = np.random.random(len(gs_circ.all_paras)) - 0.5
  
  # Get optimized result
  gs_res = minimize(
    gs_func,
    init_amp,
    args=(gs_grad_ops,),
    method=config['optim'],
    jac=True,
    tol=config['tol'],
    options={
      'maxiter': config['maxiter'], 
      'disp': config.get('debug', False),
    },
  )

  # make param cache
  if config['dump']:
    gs_pr = dict(zip(gs_circ.params_name, gs_res.x))
    with open(fp, 'w', encoding='utf-8') as fh:
      json.dump(gs_pr, fh, indent=2, ensure_ascii=False)
 
  # Get energy, NOTE: evolve `gs_sim` to `gs_circ`
  gs_ene = run_expectaion(gs_sim, ham, gs_circ, gs_res.x)

  print('E0 energy:', gs_ene)
  return gs_sim, gs_ene, np.array(gs_res.x)


def run_es(mol:MolecularData, ham:Ham, gs_sim:QVM, config:Config, init_amp:ndarray=None) -> float:
  # Declare the excited state simulator
  if isinstance(ham, ESConserveHam):
    es_sim = ESConservation(mol.n_qubits, mol.n_electrons)
    HAM_NULL = ESConserveHam(FermionOperator(''))
  else:
    es_sim = Simulator('mqvector', mol.n_qubits)
    HAM_NULL = Hamiltonian(QubitOperator(''))
  
  # Construct excited state ansatz circuit: |ψ(λ1)>
  es_circ, _ = get_ansatz(mol, config['ansatz'], config)
  # Get the expectation and gradient calculating function: <ψ(λ1)|H|ψ(λ1)>
  es_grad_ops = es_sim.get_expectation_with_grad(ham, es_circ)
  # Get the expectation and gradient calculating function of inner product: <ψ(λ0)|ψ(λ1)> where H is I
  ip_grad_ops = es_sim.get_expectation_with_grad(HAM_NULL, es_circ, circ_left=Circuit(), simulator_left=gs_sim)

  # Define the objective function to be minimized
  def es_func(x, es_grad_ops, ip_grad_ops, beta:float) -> Tuple[float, ndarray]:
    global punish_f
    f0, g0 = es_grad_ops(x)
    f1, g1 = ip_grad_ops(x)
    # Remove extra dimension of the array
    f0, f1, g0, g1 = [np.squeeze(x) for x in [f0, f1, g0, g1]]
    # reg term: `+ beta * |f1| ** 2``, where f1 = β * <ψ(λ0)|ψ(λ1)>
    punish_f = beta * np.abs(f1) ** 2
    # grad of reg term: `+ beta * (g1' * f1 + g1 * f1')`
    punish_g = beta * (np.conj(g1) * f1 + g1 * np.conj(f1))
    if PEEK: print('es:', f0.real, 'ip:', f1.real)
    return np.real(f0 + punish_f), np.real(g0 + punish_g)
  
  # Initialize amplitudes
  if config['cont_evolve'] and init_amp is not None and len(init_amp) == len(es_circ.all_paras):
    init_amp += (np.random.random(init_amp.shape) - 0.5) * 2e-5
  else:
    init_amp = np.random.random(len(es_circ.all_paras)) - 0.5
  
  # Get Optimized result: min. E1 = <ψ(λ1)|H|ψ(λ1)> + |<ψ(λ1)|ψ(λ0)>|^2
  es_res = minimize(
    es_func,
    init_amp,
    args=(es_grad_ops, ip_grad_ops, config['beta']),
    method=config['optim'],
    jac=True,
    tol=config['tol'],
    options={
      'maxiter': config['maxiter'], 
      'disp': config.get('debug', False),
    },
  )

  # Get the energy
  es_ene = run_expectaion(es_sim, ham, es_circ, es_res.x)

  print('E1 energy:', es_ene)
  return es_ene


def ocvqe_solver(mol:MolecularData, config1:Config, config2:Config) -> float:
  ansatz1: str = config1['ansatz']
  ham = get_ham(mol, ansatz1.endswith('QP'))

  # Ground state E0: |ψ(λ0)>
  gs_sim, gs_ene, init_amp = run_gs(mol, ham, config1)

  # Excited state E1 should be lower than E0
  es_ene = gs_ene + 1
  # Reg term should be small enough
  global punish_f
  EPS = config2['eps']
  punish_f = 1e5 if EPS is not None else None

  # retry on case failed
  while gs_ene >= es_ene or (EPS is not None and punish_f > EPS):
    # Excited state E1: |ψ(λ1)>
    es_ene = run_es(mol, ham, gs_sim, config2, init_amp)
    # double the reg_term coeff
    config2['beta'] *= 2

    print('reg_term:', punish_f)

  return es_ene
