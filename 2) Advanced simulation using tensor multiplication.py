#!/usr/bin/env python
# coding: utf-8

# In[145]:


import numpy as np
import time
import matplotlib.pyplot as plt

# Define single-qubit gates
X = np.array([[0, 1], [1, 0]])   # Pauli-X Gate
H = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]])  # Hadamard Gate

# Define CNOT Gate for two qubits
CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
]).reshape(2, 2, 2, 2)  # It is basically a identity tensor with the last two column flipped
#print (CNOT) to look how CNOT looks like inititally after reshapping

# Initialize n-qubit quantum state tensor |0...0>
def initial_state_tensor(n):  # It creates an n-dimensional array with shape(2,2,...,2), where each dimension has size 2.
    state = np.zeros([2] * n) # Initially we are making each element of each array as zero
    state[(0,) * n] = 1  # Making the 1st element of 1st array as 1 thus forming a n-qubit zero quantum state tensor   
    return state

# Apply a single-qubit gate to a specific qubit in an n-qubit system using tensordot
def apply_single_qubit_gate_tensor(gate, state, qubit_index):
    axes = [[1], [qubit_index]]  # specifies the axes to contract
    new_shape = list(state.shape)   # Converted the state into a list 
    new_shape[qubit_index] = 2  # ensures that the dimension corresponding to qubit_index is restored to 2, so that the final tensor matches the intended n-qubit state shape.
    return np.tensordot(gate, state, axes=axes).reshape(new_shape)   # The reshaped contracted tensor is returned

# Apply a two-qubit gate to specific qubits (control, target) in an n-qubit system
def apply_two_qubit_gate_tensor(gate, state, control, target):
    # Sort qubit indices
    if control > target:
        control, target = target, control
    # Use tensordot to contract along both control and target qubits
    axes = ([2, 3], [control, target])
    # Shape adjustment after applying the gate same as before
    new_shape = list(state.shape)
    new_shape[control] = new_shape[target] = 2
    return np.tensordot(gate, state, axes=axes).reshape(new_shape)
    
# Track the runtime as a function of the number of qubits
qubit_counts = range(1, 28)  # One can adjust according to computational limits
runtimes = []
for n in qubit_counts:
    # Initialize the state tensor
    state = initial_state_tensor(n)
    start_time = time.perf_counter()
    
    # Applying same sequence of gates as an example as before
    if n >= 1:
        state = apply_single_qubit_gate_tensor(H, state, 0)  # Apply H to the first qubit
    if n >= 2:
        state = apply_two_qubit_gate_tensor(CNOT, state, 0, 1)  # Apply CNOT to the first and second qubits
    if n >= 1:
        state = apply_single_qubit_gate_tensor(X, state, 0)  # Apply X to the first qubit again
    end_time = time.perf_counter()
    
    # Record runtime
    runtimes.append(end_time - start_time)

# Plotting runtime as a function of number of qubits
plt.plot(qubit_counts, runtimes, marker='o')
plt.xlabel("Number of Qubits")
plt.ylabel("Runtime (seconds)")
plt.title("Quantum Circuit Simulation Runtime with Tensor Operations")
plt.grid()
plt.show()

