import qgml
import matplotlib.pyplot as plt
from qml_benchmarks.data.spin_blobs import *

import numpy as np
import os
import yaml

np.random.seed = 61

###### SETTINGS #######
n_spins = 16
n_blobs = 20
num_samples_train = 5000
num_samples_test = 50000
peak_weight = 8 #average hamming weight of peak configs
noise_prob = 0.05
### END OF SETTINGS ###

current_dir = os.path.dirname(__file__)
name = f'{n_spins}_spins_{n_blobs}_blobs'
if not os.path.exists(current_dir+'/'+name):
    os.makedirs(current_dir+'/'+name)

settings = {'n_spins': n_spins, 'n_blobs': n_blobs, 'num_samples_train': num_samples_train, 'num_samples_test': num_samples_test,
            'peak_weight': peak_weight, 'noise_prob': noise_prob}

with open(current_dir+'/'+ name + '/settings.yaml', 'w') as file:
    # Dump the dictionary to the file with custom formatting
    yaml.dump(settings, file, default_flow_style=False, sort_keys=False)

def distance_sampler():
    # used to add noise to the peak spin configs.
    return np.random.binomial(n_spins, noise_prob)

# generate peak configs with low weight
p_flip = 1 / n_spins * peak_weight
peak_spins = []
while len(peak_spins) < n_blobs:
    bitstring = list((-1) ** np.random.binomial([1] * n_spins, [p_flip] * n_spins))
    if bitstring not in peak_spins:
        peak_spins.append(bitstring)

np.savetxt(current_dir+'/'+name+ f'/{name}_peak_spins.csv', peak_spins,  delimiter=',', fmt='%d')

sampler = RandomSpinBlobs(N=n_spins, num_blobs=n_blobs, distance_sampler=distance_sampler,
                          peak_spins=peak_spins)

X, y = sampler.sample(num_samples_train+num_samples_test, return_labels=True)
X = -X

np.savetxt(current_dir+'/'+name+ f'/{name}_train.csv', X[:num_samples_train], delimiter=',', fmt='%d')
np.savetxt(current_dir+'/'+name+ f'/{name}_labels_train.csv', y[:num_samples_train], delimiter=',', fmt='%d')
np.savetxt(current_dir+'/'+name+ f'/{name}_test.csv', X[-num_samples_test:], delimiter=',', fmt='%d')
np.savetxt(current_dir+'/'+name+ f'/{name}_labels_test.csv', y[-num_samples_test:], delimiter=',', fmt='%d')



