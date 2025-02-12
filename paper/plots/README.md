# Plots
Here you can find all the scripts used to calculate and plot all the relevant metrics of this paper. They all have an initial SETTINGS part, where the relevant variables are set. The parameters used to define and evaluate each model are taken from the trained models (paper/training/trained_parameters). See also the requirements.txt, with all the necessary packages to run the scripts.

## plots_config
Here we define the common variables amongst all the plot scripts (visual plot settings, directory paths...). Change it once and everything will be affected. Notice the "formatter_setup" parameter, which automatically defines the dimension and proportions of the figure according to the type of latex document you want to add them to (from the package [rsmf](https://github.com/johannesjmeyer/rsmf))


## sample_and_save
This script samples the specified models for the given dataset and saves each result in the "samples" folder. You can define a numpy seed, the number of samples for each model ("num_samples"), the models and datasets you want to make the calculation for and the parameters "num_steps_multiplier" and "max_chunk_size" coming from the classical energy based models from the repo [qml-benchmarks](https://github.com/XanaduAI/qml-benchmarks/blob/generative_models/src/qml_benchmarks/models/energy_based_model.py).

## calc_mmd_loss, calc_kgel, calc_cov_matrix
Here we calculate the MMD loss values and the KGEL for different sigmas and calculate the covariance matrices respectively, saving the results into the "tables" or "cov_matrix" folders. We do this for each of the specified models and datasets in SETTINGS. Here, we also define a numpy seed and a jax key, as well as parameters like "n_ops", "n_samples", "max_batch_ops" and "max_batch_samples" necessary for the IQP functions. For the MMD loss script we also find the parameter "std_repeats", which refers to the number of times we will calculate the MMD loss with a fresh set of samples in order to estimate its standard deviation.

## plot_mmd_loss, plot_kgel, plot_cov_matrix
In these scripts we plot the results obtained in the previous section and we save the images in the "figures" folder. The stand out parameters here are the "aspect_ratio" and the "squishing_factor" (only in plot_mmd_loss) which sets a separation between the scattered points in different models but same sigma value (the lower, the more separated).
