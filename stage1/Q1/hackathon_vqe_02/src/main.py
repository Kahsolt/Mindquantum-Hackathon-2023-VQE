# Example program. (No meaningful result)
# The judging program 'eval.py' will call `Main.run()` function from 'src/main.py',
# and receive an energy value. So please put the executing part of your algorithm
# in `Main.run()`, return an energy value.
# All of your code should be placed in 'src/' folder.

from typing import Tuple, List, Union

import numpy as np
from numpy import ndarray
from scipy.optimize import minimize
from openfermion.chem import MolecularData
from mindquantum.core.gates import H, X, Y, Z, RX, RY, RZ
from mindquantum.core.circuit import Circuit, UN
from mindquantum.simulator import Simulator
from mindquantum.core.operators import FermionOperator, InteractionOperator, normal_ordered
from mindquantum.core.operators import Hamiltonian, QubitOperator
from mindquantum.core.operators import TimeEvolution
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
# monkey-patching
ESConservation._check_qubits_max = lambda *args, **kwargs: None

# settings
ANSATZ  = 'QUCC'
OPTIM   = 'trust-constr'
TOL     = 1e-6
BETA    = 8
EPS     = 1e-3
MAXITER = 500
DEBUG   = False

if ANSATZ == 'QUCC':
    TROTTER = 1
elif ANSATZ == 'UCC':
    TROTTER = 1
elif ANSATZ == 'UCCSD':
    THRESH = 0
elif ANSATZ == 'HEA':
    DEPTH = 5
    ROT_GATES = [RX, RY]
    ENTGL_GATE = X

ANSATZS = [
    # mindquantum
    'QUCC',             # nice~
    'UCC',              # error very large ~0.1
    'UCCSD',            # error large ~0.01
    'HEA',              # FIXME: this does not work, really...?!!
    # qupack
    'UCCSD-QP',         # error very large ~0.1
]
OPTIMS = [
    'trust-constr',     # nice~
    'CG',               # error large ~0.01
    'BFGS',             # error large ~0.01
    'COBYLA',           # error very large ~1
]

if 'globals':
    reg_term = None      # save current reg_term for check


def get_ansatz(mol:MolecularData) -> Tuple[Circuit, List[float]]:
    # Construct hartreefock wave function circuit: |0...> -> |1...>
    # NOTE: this flip is important though do not know why; H does not work
    hartreefock_wfn_circuit = UN(X, mol.n_electrons)

    initial_amplitudes = None
    if ANSATZ == 'UCC':
        ansatz_circuit = UCCAnsatz(mol.n_qubits, mol.n_electrons, trotter_step=TROTTER).circuit
    elif ANSATZ == 'QUCC':
        ansatz_circuit = QubitUCCAnsatz(mol.n_qubits, mol.n_electrons, trotter_step=TROTTER).circuit
    elif ANSATZ == 'UCCSD':
        if not 'sugar':
            ansatz_circuit, initial_amplitudes, _, _, _, _ = generate_uccsd(mol, threshold=THRESH)
        else:
            ucc_fermion_ops = uccsd_singlet_generator(mol.n_qubits, mol.n_electrons, anti_hermitian=True)
            ucc_qubit_ops = Transform(ucc_fermion_ops).jordan_wigner()
            ansatz_circuit = TimeEvolution(ucc_qubit_ops.imag).circuit      # ucc_qubit_ops 中已经包含了复数因子 i 
            init_amp_ccsd = uccsd_singlet_get_packed_amplitudes(mol.ccsd_single_amps, mol.ccsd_double_amps, mol.n_qubits, mol.n_electrons)
            initial_amplitudes = [init_amp_ccsd[i] for i in ansatz_circuit.params_name]
    elif ANSATZ == 'UCCSD-QP':
        ucc_fermion_ops = uccsd_singlet_generator(mol.n_qubits, mol.n_electrons, anti_hermitian=False)
        circ = Circuit()
        for term in ucc_fermion_ops: circ += ExpmPQRSFermionGate(term)
    elif ANSATZ == 'HEA':
        ansatz_circuit = HardwareEfficientAnsatz(mol.n_qubits, ROT_GATES, ENTGL_GATE, depth=DEPTH).circuit
    else:
        raise ValueError(f'unknown ansatz: {ANSATZ}')

    if ANSATZ == 'UCCSD-QP':
        vqc = circ
    else:
        vqc = hartreefock_wfn_circuit + ansatz_circuit
    #vqc.summary()
    return vqc, initial_amplitudes


def run_gs(mol:MolecularData, ham:Union[Hamiltonian, ESConserveHam]) -> Tuple[Union[Simulator, ESConservation], float]:
    # Declare the ground state simulator
    if isinstance(ham, ESConserveHam):
        gs_sim = ESConservation(mol.n_qubits, mol.n_electrons)
    else:
        gs_sim = Simulator('mqvector', mol.n_qubits)
    # Construct ground state ansatz circuit: |ψ(λ0)>
    gs_circ, init_amplitudes = get_ansatz(mol)
    # Get the expectation and gradient calculating function: <ψ(λ0)|H|ψ(λ0)>
    gs_grad_ops = gs_sim.get_expectation_with_grad(ham, gs_circ)

    # Define the objective function to be minimized
    def gs_func(x, grad_ops) -> Tuple[float, ndarray]:
        f, g = grad_ops(x)
        f = np.squeeze(f)       # [1, 1] => []
        g = np.squeeze(g)       # [1, 1, 96] => [96]
        return np.real(f), np.real(g)
    # Initialize amplitudes
    init_amplitudes = init_amplitudes if init_amplitudes is not None else np.random.random(len(gs_circ.all_paras))
    # Get Optimized result: min. E0 = <ψ(λ0)|H|ψ(λ0)>
    gs_res = minimize(
        gs_func,
        init_amplitudes,
        args=(gs_grad_ops,),
        method=OPTIM,
        jac=True,
        tol=TOL,
        options={'maxiter':MAXITER, 'disp':DEBUG},
    )

    # Construct parameter resolver of the ground state circuit
    gs_pr = dict(zip(gs_circ.params_name, gs_res.x))
    # Calculate energy of ground state
    if isinstance(gs_sim, ESConservation):
        gs_ene = gs_sim.get_expectation(ham, gs_circ, gs_pr).real
    else:
        # Evolve into ground state
        gs_sim.apply_circuit(gs_circ, gs_pr)
        if 'sugar':
            gs_ene = gs_sim.get_expectation(ham).real
        else:
            qs0 = gs_sim.get_qs()
            H = ham.hamiltonian.matrix().todense()
            gs_ene = (qs0 @ H @ qs0).real

    print('E0 energy:', gs_ene)
    return gs_sim, gs_ene


def run_es(mol:MolecularData, ham:Union[Hamiltonian, ESConserveHam], gs_sim:Union[Simulator, ESConservation], beta:float) -> float:
    # Declare the excited state simulator
    if isinstance(ham, ESConserveHam):
        es_sim = ESConservation(mol.n_qubits, mol.n_electrons)
        HAM_NULL = ESConserveHam(FermionOperator(''))
    else:
        es_sim = Simulator('mqvector', mol.n_qubits)
        HAM_NULL = Hamiltonian(QubitOperator(''))
    # Construct excited state ansatz circuit: |ψ(λ1)>
    es_circ, init_amplitudes = get_ansatz(mol)
    # Get the expectation and gradient calculating function: <ψ(λ1)|H|ψ(λ1)>
    es_grad_ops = es_sim.get_expectation_with_grad(ham, es_circ)
    # Get the expectation and gradient calculating function of inner product: <ψ(λ0)|ψ(λ1)> where H is I
    ip_grad_ops = es_sim.get_expectation_with_grad(HAM_NULL, es_circ, circ_left=Circuit(), simulator_left=gs_sim)

    # Define the objective function to be minimized
    def es_func(x, es_grad_ops, ip_grad_ops, beta:float) -> Tuple[float, ndarray]:
        global reg_term
        f0, g0 = es_grad_ops(x)
        f1, g1 = ip_grad_ops(x)
        # Remove extra dimension of the array
        f0 = np.squeeze(f0)     # [1, 1] => []
        g0 = np.squeeze(g0)     # [1, 1] => []
        f1 = np.squeeze(f1)     # [1, 1, 96] => [96]
        g1 = np.squeeze(g1)     # [1, 1, 96] => [96]
        # reg term: `+ beta * |f1| ** 2``, where f1 = β * <ψ(λ0)|ψ(λ1)>
        punish_f = beta * np.abs(f1) ** 2
        if reg_term is not None: reg_term = np.real(punish_f)
        # grad of reg term: `+ beta * (g1' * f1 + g1 * f1')`
        punish_g = beta * (np.conj(g1) * f1 + g1 * np.conj(f1))
        return np.real(f0 + punish_f), np.real(g0 + punish_g)
    # Initialize amplitudes
    init_amplitudes = init_amplitudes if init_amplitudes is not None else np.random.random(len(es_circ.all_paras))
    # Get Optimized result: min. E1 = <ψ(λ1)|H|ψ(λ1)> + |<ψ(λ1)|ψ(λ0)>|^2
    es_res = minimize(
        es_func,
        init_amplitudes,
        args=(es_grad_ops, ip_grad_ops, beta),  # punishment coefficient
        method=OPTIM,
        jac=True,
        tol=TOL,
        options={'maxiter':MAXITER, 'disp':DEBUG},
    )

    # Construct parameter resolver of the excited state circuit
    es_pr = dict(zip(es_circ.params_name, es_res.x))
    # Calculate energy of ground state
    if isinstance(gs_sim, ESConservation):
        es_ene = es_sim.get_expectation(ham, es_circ, es_pr).real
    else:
        # Evolve into excited state
        es_sim.apply_circuit(es_circ, es_pr)
        if 'sugar':
            es_ene = es_sim.get_expectation(ham).real
        else:
            qs1 = es_sim.get_qs()
            H = ham.hamiltonian.matrix().todense()
            es_ene = (qs1 @ H @ qs1).real

    print('E1 energy:', es_ene)
    return es_ene


def excited_state_solver(mol:MolecularData) -> float:
    if ANSATZ.endswith('QP'):
        ham_of = mol.get_molecular_hamiltonian()
        inter_ops = InteractionOperator(*ham_of.n_body_tensors.values())
        ham_hiq = FermionOperator(inter_ops)
        ham_fo = normal_ordered(ham_hiq)
        ham_qo = ham_fo.real
        ham = ESConserveHam(ham_qo)     # ham of a FermionOperator
    else:
        ham_of = mol.get_molecular_hamiltonian()
        inter_ops = InteractionOperator(*ham_of.n_body_tensors.values())
        ham_hiq = FermionOperator(inter_ops)
        ham_qo = Transform(ham_hiq).jordan_wigner()     # fermi => pauli
        ham_op = ham_qo.real            # FIXME: why discard imag part?
        ham = Hamiltonian(ham_op)       # ham of a QubitOperator

    if not 'debug':
        mat = ham.hamiltonian.matrix().todense()
        diag = np.diag(mat)
        breakpoint()

    # Ground state E0: |ψ(λ0)>
    gs_sim, gs_ene = run_gs(mol, ham)
    
    global reg_term
    es_ene = gs_ene + 1
    reg_term = 1e5 if EPS is not None else None
    beta = BETA
    while gs_ene >= es_ene or (EPS is not None and reg_term > EPS):     # retry on case failed
        # Excited state E1: |ψ(λ1)>
        es_ene = run_es(mol, ham, gs_sim, beta)
        # double the reg_term coeff
        beta *= 2

        print('reg_term:', reg_term)

    return es_ene


class Main:
    def run(self, mol:MolecularData):
        return excited_state_solver(mol)
