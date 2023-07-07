# Example program. (No meaningful result)
# The judging program "eval.py" will call `Main.run()` function from "src/main.py",
# and receive an energy value. So please put the executing part of your algorithm
# in `Main.run()`, return an energy value.
# All of your code should be placed in "src/" folder.

import numpy as np
from mindquantum.core.gates import X
from mindquantum.core.circuit import Circuit
from mindquantum.simulator import Simulator
from mindquantum.core.operators import Hamiltonian
from mindquantum.algorithm.nisq import generate_uccsd
from scipy.optimize import minimize


def excited_state_solver(mol):
    uccsd_ansatz, x0, _, ham_op, n_qubits, n_electrons = generate_uccsd(mol)
    hartreefock_wfn_circuit = Circuit([X.on(i) for i in range(n_electrons)])
    circ = hartreefock_wfn_circuit + uccsd_ansatz
    sim = Simulator("mqvector", n_qubits)
    grad_ops = sim.get_expectation_with_grad(Hamiltonian(ham_op), circ)

    def func(x, grad_ops):
        f, g = grad_ops(x)
        return np.real(np.squeeze(f)), np.real(np.squeeze(g))

    res = minimize(func, x0, args=(grad_ops), method="bfgs", jac=True)
    return res.fun


class Main:
    def run(self, mol):
        return excited_state_solver(mol)
