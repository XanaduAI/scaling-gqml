import numpy as np
import jax.numpy as jnp
from iqpopt.iqp_optimizer import IqpSimulator
from jax._src.typing import Array

def z_ham_pow(ops: np.ndarray, coeffs: np.ndarray, p: int) -> list:
    """Expects a hamiltonian H (represented by "ops" and "coeffs") and returns H**p (also represented by
        a list of ops and coeffs).

    Args:
        ops (np.ndarray): Matrix with 0 and 1s. Each row represents an operator. Each 0 represents
            the identity and each 1 a given pauli operator (X, Y or Z) on the qubit of the corresponding column.
        coeffs (np.ndarray): Vector of floats. Same length as "ops" number of rows. Each coefficient
            multiplies the corresponding operator of "ops" in order to form the hamiltonian H when you sum them all.
        p (float): number of times H is multiplied by itself.

    Returns:
        list: new_tot_ops, new_tot_coef. The result of the computation with the list of operators and coefficients.
    """
    def recurrent_pow(ops: np.ndarray, coeffs: np.ndarray, p: int, tot_ops: list, tot_coef: list, res_ops: list=[], res_coef: list=[]):
        """Recurrent function that calculates H**p without simplifyng the terms that have the same operators.

        Args:
            ops (np.ndarray): Matrix with 0 and 1s. Each row represents an operator. Each 0 represents
            the identity and each 1 a given pauli operator (X, Y or Z) on the qubit of the corresponding column.
            coeffs (np.ndarray): Vector of floats. Same length as "ops" number of rows. Each coefficient
            multiplies the corresponding operator of "ops" in order to form the hamiltonian H when you sum them all.
            p (int): number of times H is multiplied by itself.
            tot_ops (list): output operators
            tot_coef (list): output coefficients
            res_ops (list, optional): Helper argument that saves the operators before multiplying them. Defaults to [].
            res_coef (list, optional): Helper argument that saves the coefficients before multiplying them. Defaults to [].
        """
        ops = np.array(ops, dtype=int)
        for i, o in enumerate(ops):
            res_ops.append(o)
            res_coef.append(coeffs[i])

            if p != 1:
                recurrent_pow(ops, coeffs, p-1, tot_ops, tot_coef, res_ops, res_coef)
            else:
                to = np.zeros_like(ops[0])
                for ro in res_ops:
                    to ^= ro
                tot_ops.append(to)
                
                tc = 1
                for rc in res_coef:
                    tc *= rc
                tot_coef.append(tc)

            res_ops.pop()
            res_coef.pop()

    tot_ops = []
    tot_coef = []

    recurrent_pow(ops, coeffs, p, tot_ops, tot_coef)

    seq = tot_ops.copy()
    arg_sort = np.array(sorted(range(len(seq)), key=lambda x: [i for i in seq[x]])).argsort()

    def gen(arr):
        for a in arr:
            yield a

    gen_sort = gen(arg_sort)
    tot_ops.sort(key=lambda x: next(gen_sort))

    gen_sort = gen(arg_sort)
    tot_coef.sort(key=lambda x: next(gen_sort))
    
    new_tot_ops = []
    new_tot_coef = []
    new_coef = 0
    for i, to in enumerate(tot_ops):
        new = True
        for nto in new_tot_ops:
            if (to == nto).all():
                new = False

        if not new:
            new_coef += tot_coef[i-1]
        else:
            if new_tot_ops != []:
                new_tot_coef.append(new_coef + tot_coef[i-1])
            new_tot_ops.append(to)
            new_coef = 0

    new_tot_coef.append(new_coef + tot_coef[i])

    new_tot_ops = np.array(new_tot_ops)
    new_tot_coef = np.array(new_tot_coef)

    return new_tot_ops, new_tot_coef


def moment_ham_exp_val_iqp(iqp_circuit: IqpSimulator, params: np.ndarray, ops: np.ndarray, coeffs: np.ndarray, moment: int,
                       n_samples: int, key: Array, indep_estimates=False) -> list:
    """Calculates the expected value of a certain moment of a Hamiltonian (represented by a list of ops and coefs).

    Args:
        iqp_circuit (IqpSimulator): The IQP circuit itself given by the class IqpSimulator.
        params (np.ndarray): The parameters of the IQP gates.
        ops (np.ndarray): Matrix with 0 and 1s. Each row represents an operator. Each 0 represents
            the identity and each 1 a given pauli operator (X, Y or Z) on the qubit of the corresponding column.
        coefs (np.ndarray): Vector of floats. Same length as "ops" number of rows. Each coefficient
            multiplies the corresponding operator of "ops" in order to form the hamiltonian H when you sum them all.
        moment (int): Moment of the expected value (number of times H is multiplied by itself).
        n_samples (int): Number of samples used to calculate the IQP expectation value.
        key (Array): Jax key to control the randomness of the process.
        indep_estimates (bool): Whether to use independent estimates of the ops in a batch (takes longer).

    Returns:
        list: expectation value of the moment and its standard deviation.
            
    """
    new_ops, new_coeffs = z_ham_pow(ops, coeffs, moment)
    mean, std = iqp_circuit.op_expval(params, new_ops, n_samples, key, indep_estimates)
    result = (new_coeffs * mean).sum()
    res_std = jnp.sqrt((new_coeffs**2 * std**2).sum())

    return result, res_std


def magnet_moment_iqp(iqp_circuit: IqpSimulator, params: np.ndarray, moment: int, n_samples: int, key: Array,
                  wires: list=None, indep_estimates=False) -> list:
    """Calculates the magnetic moment of the distribution of the circuit.

    Args:
        iqp_circuit (IqpSimulator): The IQP circuit itself given by the class IqpSimulator.
        params (np.ndarray): The parameters of the IQP gates.
        moment (int): Moment of the expected value.
        n_samples (int): Number of samples used to calculate the IQP expectation value.
        key (Array): Jax key to control the randomness of the process.
        wires (list): List of qubits positions where the operators will be measured. The rest will be traced away.
            Defaults to None, which refers to using all qubits.
        indep_estimates (bool): Whether to use independent estimates of the ops in a batch (takes longer).

    Returns:
        list: Expectation value of the moment and its standard deviation.
    """
    if wires is None:
        wires = jnp.array(range(iqp_circuit.n_qubits))
    
    ops = []
    coefs = []
    for i in range(iqp_circuit.n_qubits):
        if i in wires:
            op = np.zeros(iqp_circuit.n_qubits)
            op[i] = 1
            ops.append(op)
            coefs.append(1/iqp_circuit.n_qubits)
    ops = jnp.array(ops)
    coefs = jnp.array(coefs)
    
    return moment_ham_exp_val_iqp(iqp_circuit, params, ops, coefs, moment, n_samples, key, indep_estimates)


def energy_moment_iqp(iqp_circuit: IqpSimulator, params: np.ndarray, moment: int, j_matrix: np.ndarray, n_samples: int,
                  key: Array, ext_field: bool=False, wires: list=None, indep_estimates=False) -> list:
    """Calculates the energy moment of the distribution of the circuit.

    Args:
        iqp_circuit (IqpSimulator): The IQP circuit itself given by the class IqpSimulator.
        params (np.ndarray): The parameters of the IQP gates.
        moment (int): Moment of the expected value.
        j_matrix (np.ndarray): Interaction matrix of the spins energy.
        n_samples (int): Number of samples used to calculate the IQP expectation value.
        key (Array): Jax key to control the randomness of the process.
        ext_field (bool): If True, the energy with external field is calculated. Defaults to False.
        wires (list): List of qubits positions where the operators will be measured. The rest will be traced away.
            Defaults to None, which refers to using all qubits.
        indep_estimates (bool): Whether to use independent estimates of the ops in a batch (takes longer).

    Returns:
        list: Expectation value of the moment and its standard deviation.
    """
    if wires is None:
        wires = jnp.array(range(iqp_circuit.n_qubits))

    if len(j_matrix) != len(j_matrix[0]):
        raise Exception(f"j_matrix has to be a square matrix.")
    
    if len(wires) != len(j_matrix):
        raise Exception(f"j_matrix has {len(j_matrix)} qubits, but wires has {len(wires)} qubits.")
    
    ops = []
    coefs = []

    # External field
    if ext_field:
        for i in range(len(wires)):
            if i in wires:
                op = np.zeros(iqp_circuit.n_qubits)
                op[wires[i]] = 1
                ops.append(op)
                coefs.append(-1)

    # Interactions - j_matrix is symmetric with 0 diagonal
    for i in range(0, len(wires)-1):
        for k in range(i+1, len(wires)):
            op = np.zeros(iqp_circuit.n_qubits)
            op[wires[i]], op[wires[k]] = 1, 1
            ops.append(op)
            coefs.append(-j_matrix[i][k])
    ops = jnp.array(ops)
    coefs = jnp.array(coefs)

    return moment_ham_exp_val_iqp(iqp_circuit, params, ops, coefs, moment, n_samples, key, indep_estimates)



def magnet_moment_samples(samples: np.ndarray, moment: int) -> list:
    """Calculates the magnetic moment of the distribution given by the samples.

    Args:
        samples (np.ndarray): Samples of the distribution.
        moment (int): Moment of the expected value.

    Returns:
        list: Expectation value of the moment and its standard deviation.
    """
    samples = 1 - 2*samples
    magnets = samples.sum(axis=1)/len(samples[0])
    mag_mom = magnets**moment
    return mag_mom, jnp.mean(mag_mom), jnp.std(mag_mom)/jnp.sqrt(len(mag_mom))

def energy_moment_samples(samples: np.ndarray, moment: int, j_matrix: np.ndarray) -> list:
    """Calculates the energy moment of the distribution given by the samples

    Args:
        samples (np.ndarray): Samples of the distribution.
        moment (int): Moment of the expected value.
        j_matrix (np.ndarray): Interaction matrix of the spins energy.

    Returns:
        list: Expectation value of the moment and its standard deviation.
    """
    samples = 1 - 2*samples
    energies = - ((samples @ j_matrix) * samples).sum(axis=1) / 2
    en_mom = energies**moment
    return en_mom, jnp.mean(en_mom), jnp.std(en_mom)/jnp.sqrt(len(en_mom))