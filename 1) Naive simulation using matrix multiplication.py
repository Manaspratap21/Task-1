#!/usr/bin/env python
# coding: utf-8

# In[28]:


import numpy as np
import time
import matplotlib.pyplot as plt

# Define single-qubit gates
I = np.array([[1, 0], [0, 1]])   # Identity Gate
X = np.array([[0, 1], [1, 0]])   # Pauli-X Gate
H = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]])  # Hadamard Gate

# Initial single-qubit |0> state
zero_state = np.array([1, 0])

# Function for creating an n-qubit Quantum State /00.....0>
def initial_state(n=0):
    state =  zero_state    # Assighning the default value to be a single-qubit quantum state
    for a in range(n-1):    # Running a 'for' loop to perform the kronecker product of |0> n times with itself to build n-qubit zero qunatum state
        state=np.kron(state,zero_state)
   # print('n-qubit Quantum State is',state) if one would like to see the n-qubit Quantum State at every step
    return state

# Function to apply a single-qubit gate to the i-th qubit in an n-qubit system
def apply_single_qubit_gate(gate, state, qubit_index, n):
    # Build the full gate matrix for n qubits, with the specified gate on the `qubit_index` position
    full_gate = 1
    for i in range(n):
        if i == qubit_index:  # Locating the particular qubit index at which the operation need to be done and inserting the desired gate in the position corresponding to it
            full_gate = np.kron(full_gate, gate)
        else:   # for rest of the case inserting the identity gate so as to have no change of the qubits
            full_gate = np.kron(full_gate, I)
    # print('full gate is {}'.format(full_gate)) if one would like to see the full gate at every step
    return full_gate 

# Function to apply a CNOT gate to any two qubits (control, target) in an n-qubit system
def apply_cnot_gate(state, control, target, n):
    full_CNOT = np.zeros((2**n, 2**n), dtype=complex)  # a full CNOT matrix of size 2^n × 2^n 
    for i in range(2**n):
        binary_string_length = '0' + str(n) + 'b'  # Define the format for n-bit binary with leading zeros
        binary = format(i, binary_string_length)   # Convert integer i to binary string of length n
        if binary[control] == '1':   # if the control qubit is 1, it flips the target qubit; otherwise, the state remains the same
            flipped = binary[:target] + ('0' if binary[target] == '1' else '1') + binary[target+1:]
            j = int(flipped, 2)
            full_CNOT[i, j] = 1
        else:
            full_CNOT[i, i] = 1
    # print (full_CNOT) if one would like to see the CNOT matrix at every step
    return full_CNOT 

# Experiment setup to measure runtime for varying numbers of qubits
n = 14  # No.of qubits. Here one can also use the input() function to input the desired number of qubits
qubit_counts = range(1,n)  # Adjusted to a smaller range due to exponential growth in memory usage
runtimes = []

for n in qubit_counts:
    state = initial_state(n) # Assighning the default value to be a single-qubit quantum state
    start_time = time.time()
    
    # Apply a sequence of gates as an example
    if n >= 1:
        state = apply_single_qubit_gate(X, state, 0, n)  # Apply X to the first qubit
    if n >= 2:
        state = apply_cnot_gate(state, 0, 1, n)          # Apply CNOT to the first and second qubits if they exist
    if n >= 1:
        state = apply_single_qubit_gate(H, state, 0, n)  # Apply H to the first qubit again 
    
    end_time = time.time()
    runtimes.append(end_time - start_time)

# Plotting runtime as a function of number of qubits
plt.plot(qubit_counts, runtimes, marker='o')
plt.xlabel("Number of Qubits")
plt.ylabel("Runtime (seconds)")
plt.title("Quantum Circuit Simulation Runtime")
plt.grid()
plt.show()


# In[ ]:




