from mindquantum.simulator import Simulator
from mindquantum.simulator.mqsim import MQSim
from mindquantum.core.gates import X, H, Z
from mindquantum.core.circuit import Circuit


sim = Simulator('mqvector', 1)
print(sim.get_qs())

sim.apply_gate(H.on(0))
print(sim.get_qs())

sim.reset()
print(sim.get_qs())

qc = Circuit()
qc += H.on(0)
qc += Z.on(0)

sim.apply_circuit(qc)
print(sim.get_qs())

sim.reset()
print(sim.get_qs())
