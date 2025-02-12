import os
import zipfile
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

np.random.seed(666)

current_dir = os.path.dirname(__file__)

def load_data_skip_columns(filename):
    # Load the data file, skipping the first two columns
    data = pd.read_csv(filename, delim_whitespace=True, header=None)

    # Drop the first two columns (index 0 and 1)
    data = data.drop(columns=[0, 1])

    # Convert the DataFrame to a NumPy array
    data_array = data.values

    return data_array

### 805

url = 'https://gitlab.inria.fr/ml_genetics/public/artificial_genomes/-/raw/29c1ef7cf242e842df4360abae2eebeec995f40e/1000G_real_genomes/805_SNP_1000G_real.hapt'
filename = current_dir+'/805_SNP_1000G_real.hapt'
status = status = os.system(f'wget -O {filename} {url}')
if status != 0:
    print('Error downloading 805_SNP_1000G_real.hapt. Check if you have the latest version of wget installed in your PC.')
data = load_data_skip_columns(filename)
X_train, X_test = train_test_split(data, test_size=1/3)
np.savetxt(current_dir+'/805_SNP_1000G_real_train.csv', X_train, fmt='%d', delimiter=',')
np.savetxt(current_dir+'/805_SNP_1000G_real_test.csv', X_test, fmt='%d', delimiter=',')

url = 'https://gitlab.inria.fr/ml_genetics/public/artificial_genomes/-/raw/29c1ef7cf242e842df4360abae2eebeec995f40e/RBM_AGs/805_SNP_RBM_AG_800Epochs.hapt.zip?inline=false'
filename = current_dir+'/805_SNP_RBM_AG_800Epochs.hapt.zip'
status = os.system(f'wget -O {filename} {url}')
if status != 0:
    print('Error downloading 805_SNP_RBM_AG_800Epochs.hapt.zip.')
with zipfile.ZipFile(filename, 'r') as zip_ref:
    zip_ref.extractall(current_dir)
data = load_data_skip_columns(current_dir+'/805_SNP_RBM_AG_800Epochs.hapt')
np.savetxt(current_dir+'/805_SNP_RBM_AG_800Epochs.csv', data, fmt='%d', delimiter=',')

url = 'https://gitlab.inria.fr/ml_genetics/public/artificial_genomes/-/raw/29c1ef7cf242e842df4360abae2eebeec995f40e/GAN_AGs/805_SNP_GAN_AG_20000epochs.hapt.zip?inline=false'
filename = current_dir+'/805_SNP_GAN_AG_20000epochs.hapt.zip'
status = os.system(f'wget -O {filename} {url}')
if status != 0:
    print('Error downloading 805_SNP_GAN_AG_20000epochs.hapt.zip.')
with zipfile.ZipFile(filename, 'r') as zip_ref:
    zip_ref.extractall(current_dir)
data = load_data_skip_columns(current_dir+'/805_SNP_GAN_AG_20000epochs.hapt')
np.savetxt(current_dir+'/805_SNP_GAN_AG_20000epochs.csv', data, fmt='%d', delimiter=',')

### 10K

url = 'https://gitlab.inria.fr/ml_genetics/public/artificial_genomes/-/raw/29c1ef7cf242e842df4360abae2eebeec995f40e/1000G_real_genomes/10K_SNP_1000G_real.hapt.zip?inline=false'
filename = current_dir+'/10K_SNP_1000G_real.hapt.zip'
status = os.system(f'wget -O {filename} {url}')
if status != 0:
    print('Error downloading 10K_SNP_1000G_real.hapt.zip.')
with zipfile.ZipFile(filename, 'r') as zip_ref:
    zip_ref.extractall(current_dir)
data = load_data_skip_columns(current_dir+'/10K_SNP_1000G_real.hapt')
X_train, X_test = train_test_split(data, test_size=1/3)
np.savetxt(current_dir+'/10K_SNP_1000G_real_train.csv', X_train, fmt='%d', delimiter=',')
np.savetxt(current_dir+'/10K_SNP_1000G_real_test.csv', X_test, fmt='%d', delimiter=',')

url = 'https://gitlab.inria.fr/ml_genetics/public/artificial_genomes/-/raw/29c1ef7cf242e842df4360abae2eebeec995f40e/RBM_AGs/10K_SNP_RBM_AG_1050epochs.hapt.zip?inline=false'
filename = current_dir+'/10K_SNP_RBM_AG_1050epochs.hapt.zip'
status = os.system(f'wget -O {filename} {url}')
if status != 0:
    print('Error downloading 10K_SNP_RBM_AG_1050epochs.hapt.zip.')
with zipfile.ZipFile(filename, 'r') as zip_ref:
    zip_ref.extractall(current_dir)
data = load_data_skip_columns(current_dir+'/10K_SNP_RBM_AG_1050epochs.hapt')
np.savetxt(current_dir+'/10K_SNP_RBM_AG_1050epochs.csv', data, fmt='%d', delimiter=',')

url = 'https://gitlab.inria.fr/ml_genetics/public/artificial_genomes/-/raw/29c1ef7cf242e842df4360abae2eebeec995f40e/GAN_AGs/10K_SNP_GAN_AG_10800Epochs.hapt.zip?inline=false'
filename = current_dir+'/10K_SNP_GAN_AG_10800Epochs.hapt.zip'
status = os.system(f'wget -O {filename} {url}')
if status != 0:
    print('Error downloading 10K_SNP_GAN_AG_10800Epochs.hapt.zip.')
with zipfile.ZipFile(filename, 'r') as zip_ref:
    zip_ref.extractall(current_dir)
data = load_data_skip_columns(current_dir+'/10K_SNP_GAN_AG_10800Epochs.hapt')
np.savetxt(current_dir+'/10K_SNP_GAN_AG_10800Epochs.csv', data, fmt='%d', delimiter=',')

url = 'https://gitlab.inria.fr/ml_genetics/public/artificial_genomes/-/raw/29c1ef7cf242e842df4360abae2eebeec995f40e/GAN_AGs/10K_SNP_WGAN_AG.hapt.zip?inline=false'
filename = current_dir+'/10K_SNP_WGAN_AG.hapt.zip'
status = os.system(f'wget -O {filename} {url}')
if status != 0:
    print('Error downloading 10K_SNP_WGAN_AG.hapt.zip.')
with zipfile.ZipFile(filename, 'r') as zip_ref:
    zip_ref.extractall(current_dir)
data = load_data_skip_columns(current_dir+'/WGAN.hapt')
np.savetxt(current_dir+'/10K_SNP_WGAN_AG.csv', data, fmt='%d', delimiter=',')


