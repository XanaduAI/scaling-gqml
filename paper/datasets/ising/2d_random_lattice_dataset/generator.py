import matplotlib.pyplot as plt
import numpy as np
import os
import numpyro
from qml_benchmarks.data.ising import IsingSpins, energy
from joblib import Parallel, delayed
import networkx as nx
import yaml

np.random.seed(666)

# Numpyro usually treats the CPU as 1 device
# You can set this variable to split the CPU for parallel processing
numpyro.set_host_device_count(8)

###### SETTINGS ########
###### !!! DO NOT CHANGE !!! ########
width = 4  # Grid dimensions
T = 3 #Temperature
burn_in = 50000
num_samples_per_chain = 100000
num_chains = 8
num_samples_train = 5000 #number of training samples to generate
num_samples_test = 50000 #used for witness points and energy/magnetisation moments
random = True  #if False, Hamiltonian weights are all set to 1. If True, weights are set to random positive numbers.
###### END OF SETTINGS ########

N = width*width #num spins

#create a random 2D lattice lattice graph
G = nx.grid_2d_graph(width, width, periodic=True)
J = nx.adjacency_matrix(G).toarray()
if random:
    J = J*np.random.rand(*J.shape)*2 #random positive weights
b = np.zeros(N)

total_samples = num_samples_per_chain*num_chains
#thin the samples as much as possible
thinning = total_samples//(num_samples_train+num_samples_test)
print('chain thinning: '+str(thinning))

current_dir = os.path.dirname(__file__)
name = f'ising_{width}_{width}_T_{T}'

settings = {'width': width, 'T': T, 'burn_in': burn_in, 'num_samples_per_chain': num_samples_per_chain,
            'num_chains': num_chains, 'num_samples_train': num_samples_train, 'num_samples_test': num_samples_test}

with open(current_dir+'/'+'settings.yaml', 'w') as file:
    # Dump the dictionary to the file with custom formatting
    yaml.dump(settings, file, default_flow_style=False, sort_keys=False)

all_samples = []
all_magnetizations = []
all_energies = []

model = IsingSpins(N=N, J=J, b=b, T=T)
all_samples = model.sample(num_samples_per_chain, num_chains=num_chains, thinning=thinning, num_warmup=burn_in)
all_samples = all_samples*2-1 #convert to pm1

for sample in all_samples:
    all_magnetizations.append(np.mean(sample))
    all_energies.append(energy(sample, J, b))

all_magnetizations = np.array(all_magnetizations)
all_energies = np.array(all_energies)

#shuffle the data to remove correlations (takes a while)
idxs = np.random.permutation(all_samples.shape[0])

samples_train = all_samples[idxs[:num_samples_train]]
samples_test = all_samples[idxs[num_samples_train:num_samples_train+num_samples_test]]
magnetizations_train = all_magnetizations[idxs[:num_samples_train]]
energies_train = all_energies[idxs[:num_samples_train]]
magnetizations_test = all_magnetizations[idxs[num_samples_train:num_samples_train+num_samples_test]]
energies_test = all_energies[idxs[num_samples_train:num_samples_train+num_samples_test]]

#save as binary data
np.savetxt(current_dir+'/'+name+'_train.csv', (samples_train+1)//2, delimiter=",", fmt='%d')
np.savetxt(current_dir+'/'+name+'_test.csv', (samples_test+1)//2, delimiter=",", fmt='%d')

fig, axes = plt.subplots(ncols=5, nrows=5, tight_layout=True)
count=0
for i in range(5):
    for j in range(5):
        axes[i,j].xaxis.set_visible(False)
        axes[i,j].yaxis.set_visible(False)
        axes[i,j].imshow(np.reshape(samples_train[count],(width, width)))
        count+=1

plt.savefig(current_dir+'/'+name+'_examples.png')
plt.show()

#compute moments of energy, magneitsation
mom_en = []
mom_mag = []
for mom in range(1,6):
    mom_en.append(np.mean(energies_test**mom))
    mom_en.append(np.std(energies_test ** mom))
    mom_mag.append(np.mean(magnetizations_test**mom))
    mom_mag.append(np.std(magnetizations_test ** mom))

header = ",".join(["mom1_en", "mom1_en_std", "mom2_en", "mom2_en_std", "mom3_en", "mom3_en_std",
                   "mom4_en", "mom4_en_std", "mom5_en", "mom_en_5_std",
                   "mom1_mag", "mom1_mag_std", "mom2_mag", "mom2_mag_std", "mom3_mag", "mom3_mag_std",
                   "mom4_mag", "mom4_mag_std", "mom5_mag", "mom5_mag_std"])

np.savetxt(current_dir+'/'+name+'_moments.csv', np.array([mom_en+mom_mag]), delimiter=",", header = header)

n_bins = 30
#plot the energy and magnetization distributions
plt.hist(energies_test, bins=n_bins)
plt.title('energies test')
plt.savefig(current_dir+'/'+name+'_en_dist_test.png')
plt.show()
plt.hist(energies_train, bins=n_bins)
plt.title('energies train')
plt.savefig(current_dir+'/'+name+'_en_dist_train.png')
plt.show()
plt.hist(magnetizations_test, bins=n_bins)
plt.title('magnetizations test')
plt.savefig(current_dir+'/'+name+'_mag_dist_test.png')
plt.show()
plt.hist(magnetizations_train, bins=n_bins)
plt.title('magnetizations train')
plt.savefig(current_dir+'/'+name+'_mag_dist_train.png')
plt.show()
