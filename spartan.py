from quantum import MultiQAUM
from classical import train_validate

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from optax import adam

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

np.random.seed(0)

model = MultiQAUM(2, 8, 2)

train_X, validate_X, train_Y, validate_Y = fetch_data_random_seed_val(5, 0)

init_weights = model.initialise_weights(seed=0)
prog = model.prog(init_weights, train_X[0])
job = prog.to_circ().to_job(observable=model.obs)
job.dump("test.job")