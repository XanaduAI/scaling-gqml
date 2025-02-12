import matplotlib.pyplot as plt
import numpy as np
import rsmf
import bench_config as config
import csv

formatter = rsmf.setup(config.formatter_setup)
fig = formatter.figure(aspect_ratio=config.aspect_ratio)

labels = []
for func in config.funcs:
    for v in config.v_lines:
        if len(config.v_lines) > 1:
            labels.append(f"{config.s_lines} = {v}")
        else:
            labels.append(f"{func}")

step = (config.max-config.min)//(config.n_points-1)
loop = range(config.min, config.max + step//2, step)

if config.evaluating == "Time":

    total_times = []
    with open(config.bench_folder / f"Data/{config.filename}.csv", 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            total_times.append([float(cell) for cell in row])

    # total_times = np.loadtxt(config.bench_folder / f"Data/{config.filename}.csv", delimiter=",")
    if len(total_times) == 1:
        total_times = np.array([total_times])

    print(total_times)
    
    for i, times in enumerate(total_times):

        loop = np.array(loop[:len(times)])

        print(loop)
        times = np.array(times)

        series = np.polynomial.polynomial.Polynomial.fit(loop, np.array(times), 1)
            
        plt.scatter(loop, times)
        if config.variable not in ["max_batch_ops", "max_batch_samples"]:
            plt.plot(loop, series(loop), label=labels[i])

    plt.xlabel(config.variable)
    plt.ylabel("Time [s]")
    plt.yscale("log")
    
    if len(config.v_lines) > 1:
        plt.legend()

if config.evaluating == "Precision":

    total_precision = np.loadtxt(config.bench_folder / f"Data/{config.filename}.csv", delimiter=",")
    if len(total_precision.shape) == 1:
        total_precision = np.array([total_precision])
    
    for i, precision in enumerate(total_precision):

        loop = np.array(loop)
        precision = np.array(precision)
        
        series = np.polynomial.polynomial.Polynomial.fit(1/np.sqrt(loop), precision, 1)
        
        plt.scatter(loop, precision)
        plt.plot(loop, series(1/np.sqrt(loop)), label=labels[i])

    plt.xlabel(config.variable)
    plt.ylabel("Precision")
    plt.yscale("log")
    plt.legend()


plt.title(config.filename)
plt.savefig(config.bench_folder / f"Figures/{config.filename}.pdf", bbox_inches="tight", dpi=500)
