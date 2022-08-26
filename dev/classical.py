import numpy as np
import jax.numpy as jnp
import jax
import optax

from qat.qpus import get_default_qpu
from quantum import measure

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
        prog = model.program(weights, features)
        job = prog.to_circ().to_job(observable=model.obs)
        expec = get_default_qpu().submit(job).value
        prediction = qubit_probability(expec)
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

def validate(model, weights, val_data):
    validate_X, validate_Y = val_data
    val_acc = accuracy(model, weights, (validate_X, validate_Y))
    return val_acc

def accuracy(model, weights, data):
    errors = 0
    for x, y in zip(data[0], data[1]):
        probofone = qubit_probability(measure(model, weights, x))
        errors += abs(y - round(probofone))
    acc = 1 - errors/len(data[0])
    return acc

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