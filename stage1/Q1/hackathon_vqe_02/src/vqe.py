#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/08/02 

# implementation of original VQE and FSM in "A variational eigenvalue solver on a quantum processor"

import json
from .common import *


def run(mol:MolecularData, ham:Ham, config:Config) -> Tuple[QVM, float, Params]:
  # Declare the simulator
  sim = get_sim(mol, ham)
  # Construct ansatz circuit: ψ(λ)
  circ, init_amp = get_ansatz(mol, config['ansatz'], config)

  # Load cached params if exists
  if config.get('dump', False):
    fp = CACHE_PATH / f'vqe-{config["ansatz"]}-{Path(mol.filename).name}.json'
    if fp.exists():
      print('>> try using cached E0 states')
      try:
        with open(fp, 'r', encoding='utf-8') as fh:
          pr = json.load(fh)
        ene = run_expectaion(sim, circ, pr)
        print('E0 energy:', ene)
        return sim, ene, np.asarray(list(pr.values()))
      except:
        print('>> cache file error')
        fp.unlink()
  
  # Get the expectation and gradient calculating function: <ψ(λ)|H|ψ(λ)>
  grad_ops = sim.get_expectation_with_grad(ham, circ)

  # Define the objective function to be minimized
  def func(x:ndarray, grad_ops:Callable, hparam:Config) -> Tuple[float, ndarray]:
    f, g = grad_ops(x)
    f, g = [np.squeeze(x) for x in [f, g]]
    if PEEK: print('gs:', f.real)
    return np.real(f), np.real(g)
  
  # Get optimized results
  params = optim_scipy(func, init_amp, (grad_ops,), config)

  # Make params cache
  if config.get('dump', False):
    pr = dict(zip(circ.params_name, params))
    with open(fp, 'w', encoding='utf-8') as fh:
      json.dump(pr, fh, indent=2, ensure_ascii=False)
 
  # Get energy, evolve `sim` to `circ`
  ene = run_expectaion(sim, ham, circ, params)

  print('E0 energy:', ene)
  return sim, ene, params


def vqe_solver(mol:MolecularData, config:Config) -> float:
  ansatz: str = config['ansatz']
  ham = get_ham(mol, ansatz.endswith('QP'))

  # Ground state E0: |ψ(λ0)>
  _, gs_ene, _ = run(mol, ham, config)

  return gs_ene


def fsm_solver(mol:MolecularData, config:Config) -> float:
  ''' NOTE: 已知 特征值λ(能量) 求 特征向量(态) '''

  ansatz: str = config['ansatz']
  ham = get_ham(mol, ansatz.endswith('QP'))

  # Make folded spectrum: |H-λ|^2
  # TODO: how to do this
  ham_hat = (ham - config['lmbd']) ** 2

  # Excited state Ek: |ψ(λk)>
  _, es_ene, _ = run(mol, ham_hat, config)

  return es_ene
