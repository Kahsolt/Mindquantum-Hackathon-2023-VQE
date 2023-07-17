#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/07/16

# The physical model of VQE seems like:
#   - a moleculer has a Hamiltionian H, in form of a **diagonal** matrix (or say can be decomposed into Pauli operators) sized 2^n_qubits
#   - energy of the moleculer in ground state and each excited state (E0, E1, etc...), are the eigenvals of H (?)
#   - use a prober |psi> to get these energy by caculating the expectation E = <psi|H|psi>/<psi|psi>
#   - use an ansatz to prepare this |psi> from a reference stats |phi>

import numpy as np
from mindquantum.core.circuit import Circuit
from mindquantum.core.operators import QubitOperator, Hamiltonian
from mindquantum.simulator import Simulator


# this is the hamiltionian that holds the state of a physical quantum object
qo = QubitOperator('Z0 Y1', 0.763)    # a 2-qubits system
mat = qo.matrix().todense()
print('qubit_op mat:')
print(mat)
ham = Hamiltonian(qo)   # do not know why need this wrapper though...

# this is an encoder to prepare U|init> -> |phi>
encoder_circ = Circuit().x(0).x(1)  # |00> -> |11>
qs = encoder_circ.get_qs()
print('|phi>:')
print(qs)

# this is an ansatz to evolve U|phi> -> |psi> 
ansatz_circ = Circuit().ry(1.2, 0).rx(-0.57, 1)
qs = ansatz_circ.get_qs()
print('|psi>:')
print(qs)

# this is a quantum computer
sim = Simulator('mqvector', qo.count_qubits())
qs = sim.get_qs()    # here init state |00>
print('|init>:')
print(qs)

# we apply the evolutional ansatz
sim.apply_circuit(encoder_circ) # |init> -> |phi>
sim.apply_circuit(ansatz_circ)  # |phi> -> |psi>
qs = sim.get_qs()    # got the final state |psi>
print('|psi>:')
print(qs)

# then measure the expectation E = <psi|H|psi> 
# NOTE: we ignore the denominator, because |psi> is a pure state, hence <psi|psi> is always unit
# ref: https://www.mindspore.cn/mindquantum/docs/zh-CN/r0.8/quantum_measurement.html?highlight=get_expectation
E = sim.get_expectation(ham)
print('E:')
print(E)

# manally calc E = <psi|H|psi>
breakpoint()
E_hat = qs @ mat @ qs
print('E_hat:')
print(E_hat)

# but what does this do? 
#sim.apply_hamiltonian(ham)
#q = sim.get_qs()
#print(q)
