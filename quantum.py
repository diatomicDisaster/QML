import abc

from qat.lang.AQASM import H, RZ, RX, RY, CNOT, Program, QRoutine, build_gate
from qat.core import Observable, Term
from qat.qpus import get_default_qpu

import numpy as np

from multiprocessing import Pool
from functools import partial

@build_gate("XX", [float], arity=2)
def XX(theta):
    routine = QRoutine()
    routine.apply(H, 0)
    routine.apply(H, 1)
    routine.apply(CNOT, 0, 1)
    routine.apply(RZ(theta), 1)
    routine.apply(CNOT, 0, 1)
    routine.apply(H, 0)
    routine.apply(H, 1)
    return routine

@build_gate("YY", [float], arity=2)
def YY(theta):
    routine = QRoutine()
    routine.apply(RX(np.pi/2), 0)
    routine.apply(RX(np.pi/2), 1)
    routine.apply(CNOT, 0, 1)
    routine.apply(RZ(theta), 1)
    routine.apply(CNOT, 0, 1)
    routine.apply(RX(-np.pi/2), 0)
    routine.apply(RX(-np.pi/2), 1)
    return routine

@build_gate("ZZ", [float], arity=2)
def ZZ(theta):
    routine = QRoutine()
    routine.apply(CNOT, 0, 1)
    routine.apply(RZ(theta), 1)
    routine.apply(CNOT, 0, 1)
    return routine

@build_gate("CY", [float], arity=2)
def CY(theta):
    routine = QRoutine()
    routine.apply(RY(theta), 1)
    routine.apply(CNOT, 0, 1)
    routine.apply(RY(-theta), 1)
    routine.apply(CNOT, 0, 1)
    return routine

def measure(model, weights:np.ndarray, features:np.ndarray):
    """Perform measurement on the parameterised quantum model"""
    prog = model.program(weights, features)
    job = prog.to_circ().to_job(observable=model.obs)
    expec = get_default_qpu().submit(job).value
    return expec

class QuantumModel(abc.ABC):
    @abc.abstractmethod
    def model(self, weights, features, qbits):
        return

    def program(self, weights:np.ndarray, features:np.ndarray):
        """Return a program for the model with a set of weights and features"""
        prog = Program()
        qbits = prog.qalloc(self.nqbits)
        self.model(weights, features, qbits)
        return prog

    def _parameter_shift(self, index, weights, features):
        """Parameter-shift gradient calculation for a quantum model"""
        shift = np.zeros_like(weights)
        shift[index] += np.pi/2
        gradient = .5 * (
            measure(self, weights + shift, features) 
            - measure(self, weights - shift, features)
        )
        return gradient

    def grad(self, weights, features, grad_func):
        """Vector gradient with respect to all weights in a model"""
        gradient = []
        for index in range(len(weights)):
            gradient.append(grad_func(index, weights, features))
        return gradient

    def pool_grad(self, weights, features, grad_func):
        """Vector gradient with respect to all weights in a model"""
        with Pool(8) as pool:
            gradient = pool.map(
                partial(
                    grad_func, 
                    weights=weights, 
                    features=features
                ), 
                list(range(len(weights)))
            )
        return gradient

    def gradient(self, weights, features, index=None, method='shift', pool=True):
        """Gradient of a quantum model"""
        if method == 'shift':
            gradient_func = self._parameter_shift
        if index:
            return gradient_func(index, weights, features)
        else:
            if pool:
                return self.pool_grad(weights, features, gradient_func)
            else:
                return self.grad(weights, features, gradient_func)

class MultiQAUM(QuantumModel):

    n_local = 2
    n_entangle = 1

    def __init__(self, depth:int, nfeatures:int, nqbits:int, qpu=None):
        if nfeatures % nqbits != 0:
            raise ValueError("nqbits must be a factor of nfeatures")
        self.depth = depth
        self.nfeatures = nfeatures
        self.nqbits = nqbits
        self.features_per_qbit = int(nfeatures/nqbits)
        self.obs = Observable(nqbits, pauli_terms=[Term(1, 'Z', [0])])

    def local_layer(self, layer_weights, qbits):
        "Trainable rotation layer composed of a Z, X and Y rotation"
        for i in range(self.nqbits):
            RX(float(layer_weights[i*self.n_local]))(qbits[i])
            RY(float(layer_weights[i*self.n_local+1]))(qbits[i])
        return

    def entangling_layer(self, layer_weights, qbits):
        "Entangling layer composed of ZZ gates"
        for i in range(0, self.nqbits):
            ctr_i = i
            trg_i = 0 if i==self.nqbits-1 else i+1
            ZZ(float(layer_weights[i*self.n_entangle]))(qbits[ctr_i], qbits[trg_i])
        return

    def feature_layer(self, layer_features, qbits):
        "Embed a feature on each qubit"
        for i in range(self.nqbits):
            RZ(float(layer_features[i]))(qbits[i])
        return

    def model(self, weights, features, qbits):
        "Function to build a parameterised QAUM model on an array of qubits"
        
        for qbit in qbits: H(qbit)
        self.local_layer(
            weights[:self.n_local*self.nqbits], 
            qbits
        )
        self.entangling_layer(
            weights[
                self.n_local*self.nqbits:
                self.n_local*self.nqbits + self.n_entangle*self.nqbits
            ], 
            qbits
        )

        for i in range(self.depth):
            for j in range(self.features_per_qbit):

                layer_features = features[j * self.nqbits : (j + 1) * self.nqbits]
                self.feature_layer(layer_features, qbits)

                w = (
                    (self.n_local*self.nqbits + self.n_entangle*self.nqbits) # weights per trainable layer
                    * (self.features_per_qbit*i + j) # no. of previous trainable layers
                    + (self.n_local + self.n_entangle)*self.nqbits # zero-th weight layer
                )

                self.local_layer(weights[w:w+self.n_local*self.nqbits], qbits)
                w += self.n_local*self.nqbits
                self.entangling_layer(weights[w:w+self.n_entangle*self.nqbits], qbits)
        return

    def initialise_weights(self, seed=None):
        "Provide a uniform initialisation for the weights"
        if seed : np.random.seed(seed)
        n_train_layer = self.nqbits * self.features_per_qbit * (self.n_local + self.n_entangle)
        n = (n_train_layer*self.depth) + self.nqbits*(self.n_local + self.n_entangle)
        weights = np.random.uniform(low=-1., high=1., size=(n)) * 2.*np.pi
        return weights
