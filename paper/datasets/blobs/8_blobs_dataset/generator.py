import matplotlib.pyplot as plt
from qml_benchmarks.data.spin_blobs import generate_8blobs
import numpy as np
import os
import yaml

np.random.seed(66)

###### SETTINGS #######
num_samples_train = 5000
num_samples_test = 10000
noise_prob = 0.05
### END OF SETTINGS ###

n_spins = 16
n_blobs = 8

current_dir = os.path.dirname(__file__)
name = f'{n_spins}_spins_{n_blobs}_blobs'

settings = {'n_spins': n_spins, 'n_blobs': n_blobs, 'num_samples_train': num_samples_train, 'num_samples_test': num_samples_test,
            'noise_prob': noise_prob}

with open(current_dir+ '/settings.yaml', 'w') as file:
    # Dump the dictionary to the file with custom formatting
    yaml.dump(settings, file, default_flow_style=False, sort_keys=False)

X, y = generate_8blobs(num_samples_train+num_samples_test, noise_prob)

fig, axes = plt.subplots(ncols=20, figsize = (10,5))
for i, config in enumerate(X[:20]):
    axes[i].imshow(np.reshape(config, (4,4)))
    axes[i].set_xticklabels([])
    axes[i].set_yticklabels([])
plt.show()

np.savetxt(current_dir + f'/{name}_train.csv', X[:num_samples_train], delimiter=',', fmt='%d')
np.savetxt(current_dir + f'/{name}_labels_train.csv', y[:num_samples_train], delimiter=',', fmt='%d')
np.savetxt(current_dir + f'/{name}_test.csv', X[-num_samples_test:], delimiter=',', fmt='%d')
np.savetxt(current_dir + f'/{name}_labels_test.csv', y[-num_samples_test:], delimiter=',', fmt='%d')



