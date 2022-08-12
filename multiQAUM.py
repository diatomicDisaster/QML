import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def fetch_data_random_seed_val(n_samples, seed):
    dataset = pd.read_csv('pulsar.csv')

    data0 = dataset[dataset[dataset.columns[8]] == 0]
    data0 = data0.sample(n=n_samples, random_state=seed)
    X0 = data0[data0.columns[0:8]].values
    Y0 = data0[data0.columns[8]].values

    data1 = dataset[dataset[dataset.columns[8]] == 1]
    data1 = data1.sample(n=n_samples, random_state=seed)
    X1 = data1[data1.columns[0:8]].values
    Y1 = data1[data1.columns[8]].values

    X = np.append(X0, X1, axis=0)
    Y = np.append(Y0, Y1, axis=0)

    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, np.pi))
    X = min_max_scaler.fit_transform(X)

    # Separate the test and training datasets
    train_X, validation_X, train_Y, validation_Y = train_test_split(X, Y, test_size=0.5, random_state=seed)

    return train_X, validation_X, train_Y, validation_Y

def fetch_data_random_seed(n_samples, seed):
    dataset = pd.read_csv('pulsar.csv')

    data0 = dataset[dataset[dataset.columns[8]] == 0]
    data0 = data0.sample(n=int(n_samples / 2), random_state=seed)
    X0 = data0[data0.columns[0:8]].values
    Y0 = data0[data0.columns[8]].values

    data1 = dataset[dataset[dataset.columns[8]] == 1]
    data1 = data1.sample(n=int(n_samples / 2), random_state=seed)
    X1 = data1[data1.columns[0:8]].values
    Y1 = data1[data1.columns[8]].values
    X = np.append(X0, X1, axis=0)

    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, np.pi))
    X = min_max_scaler.fit_transform(X)

    return X, np.append(Y0, Y1, axis=0)

import abc
from qat.lang.AQASM import H, RZ, RX, RY, CNOT, Program, QRoutine, build_gate
from qat.lang.AQASM.bits import Qbit
from qat.core import Observable, Term
from qat.qpus import get_default_qpu

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
    def __init__(self, depth:int, nfeatures:int, nqbits:int, qpu=None, fpq=None):
        if nfeatures % nqbits != 0:
            raise ValueError("nqbits must be a factor of nfeatures")
        self.depth = depth
        self.nfeatures = nfeatures
        self.nqbits = nqbits
        self.features_per_qbit = fpq if fpq else int(nfeatures/nqbits)
        self.qpu = qpu if qpu else get_default_qpu()
        self.obs = Observable(nqbits, pauli_terms=[Term(1, 'Z', [0])])
        self.n_local = 2
        self.n_entangle = 1

    # def local_layer(self, layer_weights, qbits):
    #     "Trainable rotation layer composed of a Z, X and Y rotation"
    #     for i in range(self.nqbits):
    #         RZ(float(layer_weights[i*self.n_local]))(qbits[i])
    #         RX(float(layer_weights[i*self.n_local+1]))(qbits[i])
    #         RY(float(layer_weights[i*self.n_local+2]))(qbits[i])
    #     return

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
            # XX(float(layer_weights[i*self.n_entangle+1]))(qbits[ctr_i], qbits[trg_i])
            # YY(float(layer_weights[i*self.n_entangle+2]))(qbits[ctr_i], qbits[trg_i])
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

def train(model, optimizer, data, val_data=None, num_epochs=150, resume=0):
    
    train_X, train_Y = data
    if val_data: validate_X, validate_Y = val_data
    if resume == 0:
        weights = model.initialise_weights(seed=0)
        train_losses = np.zeros((num_epochs))
        train_weights = np.zeros((num_epochs+1, len(weights)))
        train_weights[0,:] = weights
    else:
        train_losses = np.loadtxt("multi_train_losses.csv", delimiter=",")
        train_weights = np.loadtxt("multi_train_weights.csv", delimiter=",")
        weights = train_weights[resume+1,:]

    opt_state = optimizer.init(weights)
    for i in range(resume, num_epochs):
        start = time.time()
        weights, opt_state, loss = step(
            model, optimizer, opt_state, weights, list(zip(train_X, train_Y)))
        end = time.time()
        train_losses[i] = loss
        train_weights[i+1,:] = weights
        np.savetxt("multi_train_weights.csv", train_weights, delimiter=',')
        np.savetxt("multi_train_losses.csv", train_losses, delimiter=',')
        train_acc = accuracy(model, weights, (train_X, train_Y))
        if val_data: 
            val_acc = accuracy(model, weights, (validate_X, validate_Y))
            print(f"Epoch {i} loss: {train_losses[i]:5.3f} Training Acc: {train_acc*100}% Validation Acc: {val_acc*100}% Time taken: {end - start:4.1f}s")
        else:
            print(f"Epoch {i} loss: {train_losses[i]:5.3f} Training Acc: {train_acc*100}% Time taken: {end - start:4.1f}s")
    np.savetxt("multi_train_losses.csv", train_losses, delimiter=',')
    np.savetxt("multi_train_weights.csv", train_weights, delimiter=',')
    return weights

def accuracy(model, weights, data):
    errors = 0
    for x, y in zip(data[0], data[1]):
        probofone = qubit_probability(model.measure(weights, x))
        errors += abs(y - round(probofone))
    acc = 1 - errors/len(data[0])
    return acc

np.random.seed(0)

from qat.core.console import display

model = MultiQAUM(2, 8, 2)

train_X, validate_X, train_Y, validate_Y = fetch_data_random_seed_val(500, 0)
#train_X, train_Y = fetch_data_random_seed(150, 0)

# init_weights = model.initialise_weights(seed=0)
# prog = model.program(init_weights, train_X[0])
# circ = prog.to_circ()
# display(circ, max_depth=None)

print(
    f"Training multiQAUM with:\n{model.depth} layers\
    \n{model.features_per_qbit} features per qubit\
    \n{model.nqbits} qubits"
)

opt = optax.adam(learning_rate=1e-1)
weights = train(model, opt, (train_X, train_Y), val_data=(validate_X, validate_Y), num_epochs=150)
