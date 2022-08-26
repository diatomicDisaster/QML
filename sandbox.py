import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from qat.qpus import get_default_qpu
from quantum import ZZQAUM

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

train_X, validate_X, train_Y, validate_Y = fetch_data_random_seed_val(5, 0)

model = ZZQAUM(2, 8, 2)
init_weights = model.initialise_weights()
prog = model.program(init_weights, train_X[0])
job = prog.to_circ().to_job(observable=model.obs)
#job.dump("test.job")

import subprocess

def submit_job(job, label):
    job.dump(f"{label}.job")
    result = subprocess.run(
        f"qat-jobrun --qpu qat.linalg.LinAlg -o {label}.res {label}.job",
        stdout=subprocess.PIPE
    )
    print(result.stdout)

submit_job(job, 'test')