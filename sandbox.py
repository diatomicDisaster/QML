from qat.lang.AQASM import Program, CNOT, QRoutine, H
from qat.qpus import get_default_qpu

qpu = get_default_qpu()

def routine_one():
    rout = QRoutine()
    qbit, anc = rout.new_wires(2)
    CNOT(qbit, anc)
    return rout

def routine_two():
    rout = QRoutine()
    qbit, anc = rout.new_wires(2)
    CNOT(qbit, anc)
    return rout

rout_one = routine_one()
rout_two = routine_two()


prog = Program()
qbits = prog.qalloc(2)
anc = prog.qalloc(1)

H(qbits[0])
H(qbits[1])
CNOT(qbits[0], qbits[1])
#CNOT(qbits[1], anc)

#prog.apply(rout_one, qbits[0], qbits[2])
#prog.apply(rout_two, qbits[1], qbits[3])

circ = prog.to_circ()
circ.display()

job = circ.to_job() #convert circuit to executable job
result = qpu.submit(job) #submit the job to a QPU

print("\nBob measures the state:")
for sample in result:
  print(f"  {sample.state} with probability {abs(sample.amplitude)**2:3.2f}") #print each complex amplitude and basis vector