import matplotlib.pyplot as plt
import numpy as np
import os
import numpyro
from qml_benchmarks.data.ising import IsingSpins, energy
from joblib import Parallel, delayed
import yaml
import networkx as nx
np.random.seed(666)

# Generates a dataset with correlations based on the Barabasi-Albert scale free graph construction

# Numpyro usually treats the CPU as 1 device
# You can set this variable to split the CPU for parallel processing
numpyro.set_host_device_count(8)

###### SETTINGS ########
N = 1000  # number of nodes
m = 2 #connectivity
T = 1 #Temperature
bias_weight = -.01
burn_in = 10000
num_samples_per_chain = 1000000
num_chains = 8
num_samples_train = 20000
num_samples_test = 20000
###### END OF SETTINGS ########

G = nx.barabasi_albert_graph(N, m)
J = nx.adjacency_matrix(G).toarray()

J = J*np.random.rand(N, N)
for i in range(J.shape[0]):
    for j in range(i):
        J[i,j] = J[j,i]

degrees = np.array([deg for node, deg in G.degree()])
b = degrees*bias_weight

total_samples = num_samples_per_chain*num_chains
#thin the samples as much as possible
thinning = total_samples//(num_samples_train+num_samples_test)
print('chain thinning: '+str(thinning))

current_dir = os.path.dirname(__file__)
name = f'ising_scale_free_{N}_nodes_T_{T}'

settings = {'n_nodes': N, 'm': m, 'T': T, 'bias_weight': bias_weight, 'burn_in': burn_in, 'num_samples_per_chain': num_samples_per_chain,
            'num_chains': num_chains, 'num_samples': num_samples_train}

with open(current_dir+'/settings.yaml', 'w') as file:
    # Dump the dictionary to the file with custom formatting
    yaml.dump(settings, file, default_flow_style=False, sort_keys=False)

all_samples = []
all_magnetizations = []
all_energies = []

model = IsingSpins(N=N, J=J, b=b, T=T, sparse=True)
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
np.savetxt(current_dir + '/' +name+'_train.csv', (samples_train+1)//2, delimiter=",", fmt='%d')
np.savetxt(current_dir + '/' +name+'_test.csv', (samples_test+1)//2, delimiter=",", fmt='%d')

# Draw the graph
plt.figure(figsize=(12, 8))
nx.draw(G, with_labels=False, node_size=50, node_color="skyblue", edge_color="gray")
plt.show()
plt.savefig(current_dir+ '/' +name+'_network.png')

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

np.savetxt(current_dir+'/'+ name +'_moments.csv', np.array([mom_en+mom_mag]), delimiter=",", header = header)

#plot the energy and magnetization distributions
plt.hist(energies_test, bins=15)
plt.title('energies test')
plt.savefig(current_dir+'/' +name+'_en_dist_test.png')
plt.show()
plt.hist(energies_train, bins=15)
plt.title('energies train')
plt.savefig(current_dir+ '/' +name+'_en_dist_train.png')
plt.show()
plt.hist(magnetizations_test, bins=15)
plt.title('magnetizations test')
plt.savefig(current_dir+ '/' +name+'_mag_dist_test.png')
plt.show()
plt.hist(magnetizations_train, bins=15)
plt.title('magnetizations train')
plt.savefig(current_dir + '/' +name+'_mag_dist_train.png')
plt.show()
