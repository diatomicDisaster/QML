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

    def measure(self, weights:np.ndarray, features:np.ndarray):
        """Perform measurement on the parameterised quantum model"""
        prog = self.program(weights, features)
        job = prog.to_circ().to_job(observable=self.obs)
        expec = self.qpu.submit(job).value
        return expec

    def _parameter_shift(self, index, weights, features):
        """Parameter-shift gradient calculation for a quantum model"""
        shift = np.zeros_like(weights)
        shift[index] += np.pi/2
        gradient = .5 * (
            self.measure(weights + shift, features) 
            - self.measure(weights - shift, features)
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
        self.qpu = qpu if qpu else get_default_qpu()
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
    
import jax.numpy as jnp
import jax
import optax

def qubit_probability(expec):
    """Convert expectation value to probability of measuring the state |1)"""
    probofone = (expec + 1.)/2 # prob. of positive prediction
    return probofone

def cross_entropy(yp, y):
    "Cross entropy loss function"
    return -y*jnp.log(yp) - (1 - y)*jnp.log(1 - yp)

def step(model, optimizer, opt_state, weights, batch):

    batch_loss = 0
    batch_grads = np.zeros_like(weights)

    for (features, label) in batch:
        prediction = qubit_probability(model.measure(weights, features))
        loss = cross_entropy(prediction, label)

        prediction_grads = model.gradient(weights, features)
        loss_grad = jax.grad(cross_entropy)(prediction, label)
        grads = loss_grad * np.array(prediction_grads)

        batch_loss += loss
        batch_grads += grads

    batch_loss /= len(batch)
    batch_grads /= len(batch)

    updates, opt_state = optimizer.update(batch_grads, opt_state, weights)
    weights = optax.apply_updates(weights, updates)

    return weights, opt_state, batch_loss

import time

def do_train(model, optimizer, data, weight, num_epochs):

    train_X, train_Y = data
    opt_state = optimizer.init(weight)
    for i in range(0, num_epochs):
        weight, opt_state, loss = step(
            model, optimizer, opt_state, weight, list(zip(train_X, train_Y))
        )
        yield weight, loss

def train(model, optimizer, data, val_data=None, num_epochs=150, resume=0):

    if resume == 0:
        init_weights = model.initialise_weights(seed=0)
        losses = np.zeros((num_epochs))
        weights = np.zeros((num_epochs+1, len(init_weights)))
        weights[0,:] = init_weights
    else:
        losses = np.loadtxt("multi_train_losses.csv", delimiter=",")
        weights = np.loadtxt("multi_train_weights.csv", delimiter=",")

    t = time.time()
    i = 0
    for weight, loss in do_train(model, optimizer, data, weights[0], num_epochs):
        t = time.time() - t
        i += 1
        weights[i+1], losses[i] = weight, loss
        np.savetxt("multi_train_weights.csv", weights, delimiter=',')
        np.savetxt("multi_train_losses.csv", losses, delimiter=',')
        train_acc = accuracy(model, weights[i+1], data)
        print(f"Epoch {i}: Loss: {losses[i]:5.3f} Training Acc: {train_acc*100}% Time taken: {t:4.1f}s")
        if val_data:
            val_acc = validate(model, weights[i+1], val_data)
            print(f"  Validation Acc: {val_acc*100}%")
    return weights

def validate(model, weights, val_data):
    validate_X, validate_Y = val_data
    val_acc = accuracy(model, weights, (validate_X, validate_Y))
    return val_acc

def accuracy(model, weights, data):
    errors = 0
    for x, y in zip(data[0], data[1]):
        probofone = qubit_probability(model.measure(weights, x))
        errors += abs(y - round(probofone))
    acc = 1 - errors/len(data[0])
    return acc