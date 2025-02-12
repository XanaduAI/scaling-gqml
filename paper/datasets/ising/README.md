To generate a dataset edit run `generate_ising_lattice.py` or `generate_ising_network.py`. 
You can edit the settings at the start of the script to generate 
different datasets. 

The script generates a few files in directory specified by the settings. 

- `XXX_examples.png`: a printout of some configurations for visual inspection (for lattice only) 
- `XXX_moments.csv`: The first 5 moments of the energy and magentization.
- `XXX_en_dist.png`: distribution of energy for visual inspection
- `XXX_mag_dist.png`: distribution of magnetization for visual inspection
- `XXX_train.csv`: The training data
- `XXX_test.csv`: The test data (used to calculate energy and mag moments)
 