from typing import List, Dict

import mindquantum

# [name suffix]
# one_one:     one   circ/sim, single thread
# one_multi:   one   circ/sim, multi  ham/thread
# multi_multi: multi circ/sim, multi  ham/thread


# <sim|circ|H|circ|sim>
def get_expectation_with_grad_one_one(
    self: mindquantum._mq_vector.mqvector, 
    arg0: mindquantum.mqbackend.hamiltonian,            # Hamiltonian<calc_type>& ham
    arg1: List[mindquantum.mqbackend.basic_gate_cxx],   # circuit_t& circ
    arg2: List[mindquantum.mqbackend.basic_gate_cxx],   # circuit_t& herm_circ
    arg3: mindquantum.mqbackend.real_pr,                # ParameterResolver& pr
    arg4: Dict[str, int],                               # MST<size_t>& p_map
  ) -> List[complex]:
  sim_l = sim_r = self
  sim_l.ApplyCircuit(circ, pr)
  sim_r.ApplyHamiltonian(ham)
  fval = dot(sim_l.qs, sim_r.qs)
  grad = get_grad(herm_circ)
  return concat([fval, grad])


# [<sim|circ|H|circ|sim>], for a list of H
def get_expectation_with_grad_one_multi(
    self: mindquantum._mq_vector.mqvector,
    arg0: List[mindquantum.mqbackend.hamiltonian],      # vector<shared_ptr<Hamiltonian<calc_type>>>& hams
    arg1: List[mindquantum.mqbackend.basic_gate_cxx],   # circuit_t& circ
    arg2: List[mindquantum.mqbackend.basic_gate_cxx],   # circuit_t& herm_circ
    arg3: mindquantum.mqbackend.real_pr,                # ParameterResolver& pr
    arg4: Dict[str, int],                               # MST<size_t>& p_map
    arg5: int,                                          # int n_thread
  ) -> List[List[complex]]:
  
  sim_l = self
  sim_l.ApplyCircuit(circ, pr)
  fval, grad = simulate_multi_thread {
    sim_r[j] = new Simulator()
    sim_r[j].ApplyHamiltonian(ham[j])
    dot(sim_l.qs, sim_r[j].qs)
  }
  return concat([fval, grad])


# [<sim|circ(θ)|H|circ(θ)|sim>], for a list of H and parametrized circ(θ)
def get_expectation_with_grad_multi_multi(
    self: mindquantum._mq_vector.mqvector, 
    arg0: List[mindquantum.mqbackend.hamiltonian],      # vector<shared_ptr<Hamiltonian<calc_type>>>& hams
    arg1: List[mindquantum.mqbackend.basic_gate_cxx],   # circuit_t& circ
    arg2: List[mindquantum.mqbackend.basic_gate_cxx],   # circuit_t& herm_circ
    arg3: List[List[float]],                            # VVT<calc_type>& enc_data
    arg4: List[float],                                  # VT<calc_type>& ans_data
    arg5: List[str],                                    # VS& enc_name
    arg6: List[str],                                    # VS& ans_name
    arg7: int,                                          # size_t batch_threads
    arg8: int,                                          # size_t mea_threads
  ) -> List[List[List[complex]]]:
  ...
  

# [<sim|circ(ψ)|H|circ(θ)|sim>], for a list of H and parametrized circ(ψ) and circ(θ)
def get_expectation_with_grad_non_hermitian_multi_multi(
    self: mindquantum._mq_vector.mqvector, 
    arg0: List[mindquantum.mqbackend.hamiltonian],      # vector<shared_ptr<Hamiltonian<calc_type>>>& hams
    arg1: List[mindquantum.mqbackend.hamiltonian],      # vector<shared_ptr<Hamiltonian<calc_type>>>& herm_hams
    arg2: List[mindquantum.mqbackend.basic_gate_cxx],   # circuit_t& left_circ
    arg3: List[mindquantum.mqbackend.basic_gate_cxx],   # circuit_t& herm_left_circ
    arg4: List[mindquantum.mqbackend.basic_gate_cxx],   # circuit_t& right_circ
    arg5: List[mindquantum.mqbackend.basic_gate_cxx],   # circuit_t& herm_right_circ
    arg6: List[List[float]],                            # VVT<calc_type>& enc_data
    arg7: List[float],                                  # VT<calc_type>& ans_data
    arg8: List[str],                                    # VS& enc_name
    arg9: List[str],                                    # VS& ans_name
    arg10: mindquantum._mq_vector.mqvector,             # derived_t& simulator_left
    arg11: int,                                         # size_t batch_threads
    arg12: int,                                         # size_t mea_threads
  ) -> List[List[List[complex]]]:
  ...

