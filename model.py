from qat.lang.AQASM import H, RZ, RX, RY, CNOT, Program, QRoutine, build_gate
from qat.core import Observable, Term
from qat.qpus import get_default_qpu

import numpy as np

from multiprocessing import Pool
from functools import partial

import optax

def cross_entropy(yp, y):
    "Cross entropy loss function"
    return -y*np.log(yp) - (1 - y)*np.log(1 - yp)

def cross_entropy_grad(yp, y):
    "Cross entropy loss function"
    return (1 - y)/(1 - yp) - y/yp

def validate(model, weights, val_data):
    validate_X, validate_Y = val_data
    val_acc = accuracy(model, weights, (validate_X, validate_Y))
    return val_acc

def accuracy(model, weights, data):
    errors = 0
    for x, y in zip(data[0], data[1]):
        probofone = expec_to_probability(measure(model, weights, x))
        errors += abs(y - round(probofone))
    acc = 1 - errors/len(data[0])
    return acc

def expec_to_probability(expec):
    """Convert expectation value to probability of measuring the state |1)"""
    probofone = (expec + 1.)/2 # prob. of positive prediction
    return probofone

def measure(model, weights:np.ndarray, features:np.ndarray):
    """Perform measurement on the parameterised quantum model"""
    prog = model.program(weights, features)
    job = prog.to_circ().to_job(observable=model.obs)
    expec = get_default_qpu().submit(job).value
    return expec

def parameter_shift(index, weights, features, model):
    """Parameter-shift gradient calculation for a quantum model"""
    shift = np.zeros_like(weights)
    shift[index] += np.pi/2
    gradient = .5 * (
        measure(model, weights + shift, features) 
        - measure(model, weights - shift, features)
    )
    return gradient

def grad(weights, features, grad_func, model):
    """Vector gradient with respect to all weights in a model"""
    gradient = []
    for index in range(len(weights)):
        gradient.append(grad_func(index, weights, features, model))
    return np.array(gradient)

def pool_grad(weights, features, grad_func, model, nthreads):
    """Vector gradient with respect to all weights in a model"""
    with Pool(nthreads) as pool:
        gradient = pool.map(
            partial(
                grad_func,
                weights=weights,
                features=features,
                model=model
            ), 
            list(range(len(weights)))
        )
    return np.array(gradient)

def gradient(model, weights, features, index=None, method='shift', pool=False):
    """Gradient of a quantum model"""
    if method == 'shift':
        gradient_func = parameter_shift
    if index:
        return gradient_func(index, weights, features, model)
    else:
        if pool:
            return pool_grad(weights, features, gradient_func, model, pool)
        else:
            return grad(weights, features, gradient_func, model)

def loss_and_grad(model, weights, features, label):
    """Calculate loss and gradient with respect to a set of weights for a given sample"""
    expec = measure(model, weights, features)
    loss = cross_entropy(expec_to_probability(expec), label)
    
    prediction_grads = gradient(model, weights, features)
    loss_grad = cross_entropy_grad(expec_to_probability(expec), label)

    grad = loss_grad * prediction_grads
    return loss, grad

def step(model, optimizer, opt_state, weights, batch):
    """Perform a step of the optimizer by applying gradients across a batch of samples"""
    batch_loss = 0
    batch_grad = np.zeros_like(weights)
    
    for (features, label) in batch:
        loss, grad = loss_and_grad(model, weights, features, label)
        
        batch_loss += loss
        batch_grad += grad
        
    updates, opt_state = optimizer.update(batch_grad/len(batch), opt_state, weights)
    weights = optax.apply_updates(weights, updates)

    return weights, opt_state, batch_loss/len(batch)

def train(self, optimizer, data, weight, num_epochs):
    train_X, train_Y = data
    opt_state = optimizer.init(weight)
    for i in range(0, num_epochs):
        weight, opt_state, loss = step(
            self, optimizer, opt_state, weight, list(zip(train_X, train_Y))
        )
        yield weight, loss

def resume_or_init(model, num_epochs, resume):
    if resume == 0:
        init_weights = model.initialise_weights(seed=0)
        losses = np.zeros((num_epochs))
        weights = np.zeros((num_epochs+1, len(init_weights)))
        weights[0,:] = init_weights
    else:
        losses = np.loadtxt("multi_train_losses.csv", delimiter=",")
        weights = np.loadtxt("multi_train_weights.csv", delimiter=",")
    return weights, losses

def train_validate(model, optimizer, data, val_data=None, num_epochs=150, resume=0):
    weights, losses = resume_or_init(model, num_epochs, resume)
    for i, (weight, loss) in enumerate(train(model, optimizer, data, weights[0], num_epochs), start=1):
        weights[i+1], losses[i] = weight, loss
        np.savetxt("multi_train_weights.csv", weights, delimiter=',')
        np.savetxt("multi_train_losses.csv", losses, delimiter=',')
        train_acc = accuracy(model, weight, data)
        print(f"Epoch {i}: Loss: {loss:5.3f} Training Acc: {train_acc*100}%")
        if val_data:
            val_acc = validate(model, weight, val_data)
            print(f"  Validation Acc: {val_acc*100}%")
    return weights

class QuantumModel:

    def __init__(self, depth:int, nfeatures:int, nqbits:int, qpu=None):
        if nfeatures % nqbits != 0:
            raise ValueError("nqbits must be a factor of nfeatures")
        self.depth = depth
        self.nfeatures = nfeatures
        self.nqbits = nqbits
        self.features_per_qbit = int(nfeatures/nqbits)
        self.obs = Observable(nqbits, pauli_terms=[Term(1, 'Z', [0])])

    def program(self, weights:np.ndarray, features:np.ndarray):
        """Return a program for the model with a set of weights and features"""
        prog = Program()
        qbits = prog.qalloc(self.nqbits)
        self.model(weights, features, qbits)
        return prog

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

@build_gate("ZZ", [float], arity=2)
def ZZ(theta):
    routine = QRoutine()
    routine.apply(CNOT, 0, 1)
    routine.apply(RZ(theta), 1)
    routine.apply(CNOT, 0, 1)
    return routine

class ZZEncoder(QuantumModel):

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
