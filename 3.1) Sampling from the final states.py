#!/usr/bin/env python
# coding: utf-8

# In[41]:


# Statevector Representation
# In quantum mechanics, measuring a quantum state collapses it into one of the basis states, and the probability of each outcome depends on the amplitude (probability amplitude) associated with each state.
# Sampling involves obtaining the probability distribution of the possible measurement outcomes and then drawing samples according to these probabilities.

import numpy as np

# Function to Compute the probabilities of each basis state
def calculating_probabilities(state_vector):
    probabilities = np.abs(state_vector) ** 2 #The probability of each basis state ∣x⟩ is given by the square of the amplitude's magnitude in the statevector. For a statevector
    return probabilities

# Function to Generate samples from the basis states based on their probabilities
def generate_samples(num_qubits,num_samples,state_vector):
    basis_states = [format(i, f'0{num_qubits}b') for i in range(2**num_qubits)]  # generates all possible basis states for an n-qubit quantum system, represented as binary strings
    samples = np.random.choice(basis_states, size=num_samples, p=calculating_probabilities(state_vector))  # Creating a list of measurement results (from basis_states) drawn according to their respective probabilities (probabilities) and simulating the effect of multiple measurements by repeating the sampling process num_samples times.
    return samples
    
# Define the number of qubits
n = 3  # Example with 3 qubits
# Example statevector of length 2^n (3 qubits -> 2^3 = 8 elements)
# This should be obtained from the output of a quantum circuit
statevector = np.array([0.5 + 0j, 0.5 + 0j, 0.5 + 0j, 0.5 + 0j,
                        0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j])

# Assuming `statevector` is a 2^n array representing the state
num_qubits = int(np.log2(len(statevector)))  # To determine the number of qubits, n, based on the length of the statevector. This formula comes from rearranging L=2^n
num_samples = 1000  # Number of measurement samples

# Output sampled measurements
print("Sampled measurement outcomes:")
print(generate_samples(num_qubits,num_samples,statevector))


# In[39]:


# Tensor Representation
# Assuming `tensor_state` is a tensor with shape (2, 2, ..., 2) for n qubits
# We flatten the n-dimensional tensor to get a 1D statevector, making it easier to compute probabilities
# The remaining steps are identical to the statevector case: we compute probabilities from the squared amplitudes and sample according to these probabilities

# Define the number of qubits
n = 3  # Example with 3 qubits

# Example tensor state for 3 qubits, shape (2, 2, 2)
# This should be obtained from the output of a quantum circuit
tensor_state = np.array([[[0.5+0j, 0.5+0j], [0.5+0j, 0.5+0j]], 
                         [[0+0j, 0+0j], [0+0j, 0+0j]]])

# Flattenning the tensor to convert it to a statevector form
statevector = tensor_state.flatten()

# Assuming `statevector` is a 2^n array representing the state
num_qubits = int(np.log2(len(statevector)))  # To determine the number of qubits, n, based on the length of the statevector. This formula comes from rearranging L=2^n
num_samples = 1000  # Number of measurement samples

# Output sampled measurements
print("Sampled measurement outcomes:")
print(generate_samples(num_qubits,num_samples,statevector))


# In[ ]:




