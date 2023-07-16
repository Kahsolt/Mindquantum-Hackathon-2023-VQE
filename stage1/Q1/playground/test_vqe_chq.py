from argparse import ArgumentParser

import numpy as np
from pychemiq import Molecules, ChemiQ, QMachineType
from pychemiq.Transform.Mapping import jordan_wigner, MappingType
from pychemiq.Optimizer import vqe_solver
from pychemiq.Optimizer import (
    DEF_NELDER_MEAD,
    DEF_POWELL,
    DEF_COBYLA,
    DEF_LBFGSB,
    DEF_SLSQP,
    DEF_GRADIENT_DESCENT,
)
from pychemiq.Circuit.Ansatz import UCC

from mol_data import GEOMETRY, get_mol_geo


def get_mol(args) -> Molecules:
    geom = get_mol_geo(args.molecule, fmt='chq')
    basis = 'sto-3g'
    multiplicity = 1
    charge = 0
    return Molecules(geom, basis, multiplicity, charge)


def run_chemiq(args, mol):
    fermion = mol.get_molecular_hamiltonian()
    print('fermion:')
    print(fermion)
    pauli = jordan_wigner(fermion)
    print('pauli:')
    print(pauli)
    pauli_size = len(pauli.data())

    machine_type = QMachineType.CPU_SINGLE_THREAD
    mapping_type = MappingType.Jordan_Wigner
    ucc_type = args.type
    n_elec = mol.n_electrons
    n_qubits = mol.n_qubits
    chemiq = ChemiQ()
    chemiq.prepare_vqe(machine_type, mapping_type, n_elec, pauli_size, n_qubits)
    ansatz = UCC(ucc_type, n_elec, mapping_type, chemiq=chemiq)

    METHODS = [
        DEF_NELDER_MEAD,
        DEF_POWELL,
        DEF_COBYLA,
        DEF_LBFGSB,
        DEF_SLSQP,
        DEF_GRADIENT_DESCENT,
    ]
    for method in METHODS:
        lr = 0.1
        init_para = np.zeros(ansatz.get_para_num()) + 1e-6
        solver = vqe_solver(method, ansatz, pauli, init_para, chemiq, lr)
        result = solver.fun_val
        n_calls = solver.fcalls
        print(f'[{method}]')
        print(f'   n_calls: {n_calls}, result: {result}')


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-M', '--molecule', default='LiH', choices=GEOMETRY.keys(), help='molecule name')
  parser.add_argument('-T', '--type', default='UCCSD', choices=['UCCS', 'UCCD', 'UCCSD'], help='ucc_type')
  args = parser.parse_args()

  mol = get_mol(args)
  run_chemiq(args, mol)
