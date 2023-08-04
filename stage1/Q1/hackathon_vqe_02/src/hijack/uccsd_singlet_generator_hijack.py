from itertools import product, combinations
from typing import List

from mindquantum.core.circuit import Circuit
from mindquantum.core.operators import FermionOperator
from mindquantum.core.operators.utils import down_index, up_index
from qupack.vqe.gates import ExpmPQRSFermionGate

from mindquantum.third_party.unitary_cc import *
assert uccsd_singlet_generator


def uccsd_singlet_generator_hijack(n_qubits, n_electrons, anti_hermitian=True, n_trotter=1) -> Circuit:
    n_spatial_orbitals = n_qubits // 2
    n_occupied = int(numpy.ceil(n_electrons / 2))
    n_virtual = n_spatial_orbitals - n_occupied

    ops: List[FermionOperator] = []
    spin_index_functions = [up_index, down_index]

    for t in range(n_trotter):
        # Generate all spin-conserving single and double excitations derived from one spatial occupied-virtual pair
        for i, (p, q) in enumerate(product(range(n_virtual), range(n_occupied))):
            # Get indices of spatial orbitals
            virtual_spatial = n_occupied + p
            occupied_spatial = q

            for spin in range(2):
                # Get the functions which map a spatial orbital index to a spin orbital index
                this_index  = spin_index_functions[spin]
                other_index = spin_index_functions[1 - spin]

                # Get indices of spin orbitals
                virtual_this   = this_index (virtual_spatial)
                virtual_other  = other_index(virtual_spatial)
                occupied_this  = this_index (occupied_spatial)
                occupied_other = other_index(occupied_spatial)

                # Generate single excitations
                coeff = ParameterResolver({f't_{t}_s_{i}': 1})
                ops += [FermionOperator(((virtual_this, 1), (occupied_this, 0)), coeff)]
                if anti_hermitian:
                    ops += [FermionOperator(((occupied_this, 1), (virtual_this, 0)), -1 * coeff)]

                # Generate double excitation
                coeff = ParameterResolver({f't_{t}_d1_{i}': 1})
                ops += [FermionOperator(((virtual_this, 1), (occupied_this, 0), (virtual_other, 1), (occupied_other, 0)), coeff)]
                if anti_hermitian:
                    ops += [FermionOperator(((occupied_other, 1), (virtual_other, 0), (occupied_this, 1), (virtual_this, 0)), -1 * coeff)]

        # Generate all spin-conserving double excitations derived from two spatial occupied-virtual pairs
        for i, ((p, q), (r, s)) in enumerate(combinations(product(range(n_virtual), range(n_occupied)), 2)):
            # Get indices of spatial orbitals
            virtual_spatial_1  = n_occupied + p
            occupied_spatial_1 = q
            virtual_spatial_2  = n_occupied + r
            occupied_spatial_2 = s

            # Generate double excitations
            coeff = ParameterResolver({f't_{t}_d2_{i}': 1})
            for (spin_a, spin_b) in product(range(2), repeat=2):
                # Get the functions which map a spatial orbital index to a
                # spin orbital index
                index_a = spin_index_functions[spin_a]
                index_b = spin_index_functions[spin_b]

                # Get indices of spin orbitals
                virtual_1_a  = index_a(virtual_spatial_1)
                occupied_1_a = index_a(occupied_spatial_1)
                virtual_2_b  = index_b(virtual_spatial_2)
                occupied_2_b = index_b(occupied_spatial_2)

                if virtual_1_a == virtual_2_b or occupied_1_a == occupied_2_b: continue

                ops += [FermionOperator(((virtual_1_a, 1), (occupied_1_a, 0), (virtual_2_b, 1), (occupied_2_b, 0)), coeff)]
                if anti_hermitian:
                    ops += [FermionOperator(((occupied_2_b, 1), (virtual_2_b, 0), (occupied_1_a, 1), (virtual_1_a, 0)), -1 * coeff)]

    circ = Circuit()
    for op in ops:
        for term in op:
            circ += ExpmPQRSFermionGate(term)
    return circ
