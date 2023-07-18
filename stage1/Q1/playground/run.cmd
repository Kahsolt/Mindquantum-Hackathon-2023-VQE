REM UCCSD ansatz in pychemiq, grid search over optimizer
python test_vqe_chq.py


REM UCCSD ansatz in mindquantum & qupack
python test_vqe_mq.py

REM UCCSD ansatz in mindquantum & qupack, grid search over optimizer & initializer
python test_vqe_mq.py --run_all_all


REM grid search for ansatz in mindquantum & qupack
python benchemark.py
