import jax
# jax.config.update('jax_default_device', jax.devices('gpu')[0])
# jax.config.update('jax_platform_name', 'cpu')
# jax.config.update('jax_enable_compilation_cache', False)
print(jax.devices())
import jax.numpy as jnp
import iqpopt as iqp
import iqpopt.gen_qml as genq
import numpy as np
import time
import bench_config as config
import csv
import scipy
import cProfile
import pstats
import os
import pickle

step = (config.max-config.min)//(config.n_points-1)
loop = range(config.min, config.max + step//2, step)

kwargs_arr = []
for v in config.v_lines:
    kwarg = config.default.copy()
    kwarg[config.s_lines] = v
    kwargs_arr.append(kwarg)

current_gate_config = [0,0]
total_times, total_precision, total_precision2 = [], [], []
for func in config.funcs:
    for kwargs in kwargs_arr:
        
        n_qubits = kwargs["n_qubits"]
        sparse = kwargs["sparse"]
        n_samples = kwargs["n_samples"]
        n_ops = kwargs["n_ops"]
        n_gates = kwargs["n_gates"]
        max_batch_samples = kwargs["max_batch_samples"]
        max_batch_ops = kwargs["max_batch_ops"]
        
        times, precision = [], []
        total_times.append([])
        total_precision.append([])
        for var in loop:

            if config.variable == "n_qubits":
                n_qubits = var
            elif config.variable == "n_samples":
                n_samples = var
            elif config.variable == "n_ops":
                n_ops = var
            elif config.variable == "n_gates":
                n_gates = var
            elif config.variable == "max_batch_samples":
                max_batch_samples = var
            elif config.variable == "max_batch_ops":
                max_batch_ops = var

            filename = f'./Data/gates_{n_qubits}_{n_gates}.pkl'
            if os.path.exists(filename):
                with open(filename, "rb") as f:
                    gates = pickle.load(f)
            else:
                gens = np.random.binomial(1, 1 / n_qubits * 3, (n_gates, n_qubits))
                gates = [[list(np.where(gens[i] > 0)[0])] for i in range(len(gens))]
                with open(filename, 'wb') as f:
                    pickle.dump(gates, f)

            params_init = np.random.uniform(0, 2*np.pi, len(gates))
            circuit = iqp.IqpSimulator(n_qubits, gates, sparse=sparse)
            key = jax.random.PRNGKey(np.random.randint(0, 99999))

            if func == "op_expval":
                jax.clear_caches()
                ops = np.random.binomial(1, 1 / n_qubits * 3, (n_ops, n_qubits))

                if not sparse:
                    ops = jnp.array(ops)
                    #if 'gpu' in config.filename:
                    #    print(jax.devices())
                    #    gpu_device = jax.devices()[0]
                    #    ops = jax.device_put(ops, device=gpu_device)
                    #    params_init = jax.device_put(params_init, device=gpu_device)


                    fun = jax.jit(circuit.op_expval_batch, static_argnames=['n_samples'])
                    if config.evaluating == "Time":
                        value, std = fun(params_init, ops, n_samples, key)

                    # cProfile.run('circuit.op_expval(params_init, ops, n_samples, key, max_batch_samples=max_batch_samples, max_batch_ops=max_batch_ops)',
                    #              './profiling.prof')
                    #
                    # with open("profile_results.txt", "w") as f:
                    #     stats = pstats.Stats('profiling.prof', stream=f)
                    #     stats.strip_dirs()
                    #     stats.sort_stats('cumulative')  # Options: 'time', 'cumulative', etc.
                    #     stats.print_stats()

                    start = time.perf_counter()
                    value, std = fun(params_init, ops, n_samples, key)
                    print(value)
                    print(std)

                    precision.append(np.mean(std))
                    values = []
                else:
                    fun = circuit.op_expval_batch
                    ops = scipy.sparse.csr_matrix(ops)

                    start = time.perf_counter()
                    value, std = fun(params_init, ops, n_samples, key)
                    precision.append(np.mean(std))
                    values = []



            elif func == "MMD loss":
                jax.clear_caches()
                ground_truth = np.random.randint(0, 2, (config.n_test, n_qubits))

                fun = genq.mmd_loss_iqp
                fun(params_init, circuit, ground_truth, 1.0, n_ops, n_samples, key,
                           max_batch_samples=max_batch_samples, max_batch_ops=max_batch_ops)

                start = time.perf_counter()
                loss = fun(params_init, circuit, ground_truth, 1.0, n_ops, n_samples, key,
                                         max_batch_samples=max_batch_samples, max_batch_ops=max_batch_ops)


            elif func == "Loss gradient":
                jax.clear_caches()
                ground_truth = np.random.randint(0, 2, (config.n_test, n_qubits))
                grad_mmd = jax.grad(genq.mmd_loss_iqp)

                grad = grad_mmd(params_init, circuit, ground_truth, config.sigma, n_ops, n_samples, key,
                                max_batch_samples=max_batch_samples, max_batch_ops=max_batch_ops)
                start = time.perf_counter()
                grad = grad_mmd(params_init, circuit, ground_truth, config.sigma, n_ops, n_samples, key, max_batch_samples=max_batch_samples, max_batch_ops=max_batch_ops)

            times.append(time.perf_counter() - start)

            total_times[-1] = times
            total_precision[-1] = precision

            save_array = total_times if config.evaluating == "Time" else total_precision

            with open(config.bench_folder / f"Data/{config.filename}.csv", 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(save_array)

        # total_times.append(times)
        # total_precision.append(precision)

save_array = total_times if config.evaluating == "Time" else total_precision
np.savetxt(config.bench_folder / f"Data/{config.filename}.csv", save_array, delimiter=",")
