import numpy as np
import pennylane as qml
from itertools import product

def penn_obs(op: np.ndarray) -> qml.operation.Observable:
    """Returns a pennylane observable from a bitstring representation.

    Args:
        op (np.ndarray): Bitstring representation of the Z operator.

    Returns:
        qml.Observable: Pennylane observable.
    """
    for i, z in enumerate(op):
        if i==0:
            if z:
                obs = qml.Z(i)
            else:
                obs = qml.I(i)
        else:
            if z:
                obs @= qml.Z(i)
    return obs

def penn_op_expval_circ(iqp_circuit, params: np.ndarray, op: np.ndarray) -> qml.measurements.ExpectationMP:
    """Defines the circuit that calculates the expectation value of the operator with the IQP circuit with pennylane tools.

    Args:
        params (np.ndarray): The parameters of the IQP gates.
        op (np.ndarray): Bitstring representation of the Z operator.

    Returns:
        qml.measurements.ExpectationMP: Pennylane circuit with an expectation value.
    """
    iqp_circuit.iqp_circuit(params)
    obs = penn_obs(op)
    return qml.expval(obs)

def penn_op_expval(iqp_circuit, params: np.ndarray, op: np.ndarray) -> float:
    """Calculates the expectation value of the operator with the IQP circuit with pennylane tools.

    Args:
        params (np.ndarray): The parameters of the IQP gates.
        op (np.ndarray): Bitstring representation of the Z operator.

    Returns:
        float: Expectation value.
    """
    dev = qml.device(iqp_circuit.device, wires=iqp_circuit.n_qubits)
    penn_op_expval_exe = qml.QNode(penn_op_expval_circ, dev)
    return penn_op_expval_exe(iqp_circuit, params, op)

def penn_x_circuit(x: np.ndarray):
    """Applies a pennylane circuit in order to initialize the qubits in the state of the bitstring x.

    Args:
        x (np.ndarray): Bitstring.
    """
    for i, b in enumerate(x):
        if b:
            qml.X(i)

def penn_train_expval(x: np.ndarray, op: np.ndarray) -> qml.measurements.ExpectationMP:
    """Defines the circuit that calculates the expectation value of the operator with the training circuit with pennylane tools.

    Args:
        x (np.ndarray): Bitstring representing the initialization of the circuit.
        op (np.ndarray): Bitstring representing the observable that we want to measure.

    Returns:
        qml.measurements.ExpectationMP: Pennylane circuit with an expectation value.
    """
    penn_x_circuit(x)
    obs = penn_obs(op)
    return qml.expval(obs)

def penn_train_expval_dev(iqp_circuit, training_set: np.ndarray, op: np.ndarray) -> float:
    """Calculates the expectation value of the operator with the training circuit with pennylane tools.

    Args:
        training_set (np.ndarray): The training set of samples from which we are trying to learn the distribution.
        op (np.ndarray): The operator that is being measured.

    Returns:
        float: The result of the expectation value.
    """
    dev = qml.device(iqp_circuit.device, wires=iqp_circuit.n_qubits)
    penn_train_expval_node = qml.QNode(penn_train_expval, dev)
    tr_train = 0
    for x in training_set:
        tr_train += penn_train_expval_node(x, op)
    tr_train /= len(training_set)
    return tr_train

def penn_mmd_loss(iqp_circuit, params: np.ndarray, training_set: np.ndarray, sigma: float) -> float:
    """Calculates the exact MMD Loss for the given IQP circuit with Pennylane tools.

    Args:
        params (np.ndarray): The parameters of the IQP gates.
        training_set (np.ndarray): The training set of samples from which we are trying to learn the distribution.
        sigma (float): Sigma parameter, the width of the kernel.

    Returns:
        float: The value of the loss.
    """
    loss = 0
    p_MMD = (1-np.exp(-1/2/sigma))/2
    for op in product([0,1], repeat=iqp_circuit.n_qubits):
        op = np.array(op)
        tr_iqp = penn_op_expval(iqp_circuit, params, op)
        tr_train = penn_train_expval_dev(iqp_circuit, training_set, op)
        loss += (1-p_MMD)**(iqp_circuit.n_qubits-op.sum()) * p_MMD**op.sum() * (tr_iqp*tr_iqp - 2*tr_iqp*tr_train + tr_train*tr_train)
    return loss



