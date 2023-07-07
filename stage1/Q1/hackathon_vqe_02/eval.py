# Judging program. You can run it to test your algorithm.

import sys
sys.path.append("./src")
import os
os.environ["OMP_NUM_THREADS"] = "4"
from src.main import Main
from openfermion.chem import MolecularData
import numpy as np
import time


class Timer:
    def __init__(self, t0=0.0):
        self.start_time = time.time()
        self.t0 = t0

    def runtime(self):
        return time.time() - self.start_time + self.t0

    def resetime(self):
        self.start_time = time.time()


# molecules and their FCI excited state energy
molecules = [
    ("H4_0.5", -0.90886229),
    ("H4_1.0", -1.93375723),
    ("H4_1.5", -1.92555851),
    ("LiH_0.5", -7.00849729),
    ("LiH_1.5", -7.7606092),
    ("LiH_2.5", -7.77815121),
#   Unknown1,
#   Unknown2,
#   Unknown3,
]
err_num = np.ones(len(molecules))

if __name__ == "__main__":
    with open("./output_info.o", "a") as f:
        main = Main()
        timer = Timer()
        en_list, time_list = [], []
        for idx, (molecule, energy) in enumerate(molecules):
            print("Start: ", molecule, file=f)
            mol_file = "./molecule_files/" + molecule
            mol = MolecularData(filename=mol_file)
            mol.load()
            t0 = timer.runtime()
            en_list.append(main.run(mol))
            time_list.append(timer.runtime() - t0)
            if abs(en_list[-1] - energy) <= 0.0016:
                err_num[idx] = 0

        if len(en_list) != len(molecules):
            print("The length of en_list is not equal to that of molecules!", file=f)
        total_time = np.sum(time_list)
        if (err_num == 1).any():
            score = err_num.sum() * 10000
        else:
            score = total_time

        print("Molecule_information: ", molecules, file=f)
        print("Result_energies: ", en_list, file=f)
        print("Time_list: ", time_list, file=f)
        print("Num_err: ", err_num, file=f)
        print("Total_time: ", total_time, file=f)
        print("Score: ", score, file=f)

        print("Score: ", score)
