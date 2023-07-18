python solver.py --run_all --log_path log.pyscf --objective pyscf
python solver.py --run_all --log_path log.uccsd --objective uccsd

python vis_fci_cmp.py --log_path log.pyscf
python vis_fci_cmp.py --log_path log.uccsd


python solver.py -O trust-constr
python solver.py -O BFGS
python solver.py -O CG
python solver.py -O COBYLA
python solver.py -O Nelder-Mead
python solver.py -O Powell
python solver.py -O TNC
python solver.py -O SLSQP


python solver.py
python vis_fci.py
