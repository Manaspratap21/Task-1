#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Statevector Representation
# In the statevector approach, we represent the quantum state as a 1D array, where each element is a complex amplitude corresponding to a basis state
# We apply the operator as a matrix multiplication and compute the inner product to get the expectation value

import numpy as np

# Define the number of qubits
n = 2  # Example with 2 qubits

# Example statevector for a 2-qubit state (should be normalized)
# This state could be obtained from the output of a quantum circuit
statevector = np.array([0.707 + 0j, 0 + 0j, 0.707 + 0j, 0 + 0j])  # |Ψ+> Bell state

# Define an example 2-qubit operator (e.g., Z ⊗ Z, which measures parity of qubits)
# The Z ⊗ Z operator is represented in the computational basis as a 4x4 matrix:
# [[ 1,  0,  0,  0],
#  [ 0, -1,  0,  0],
#  [ 0,  0, -1,  0],
#  [ 0,  0,  0,  1]]
Op = np.array([[1, 0, 0, 0],
               [0, -1, 0, 0],
               [0, 0, -1, 0],
               [0, 0, 0, 1]])

# Step 1: Apply the operator to the statevector
# This gives us Op |Ψ⟩
transformed_state = Op @ statevector  #   Matrix-vector multiplication of operator on state vector

# Step 2: Compute the inner product ⟨Ψ| Op |Ψ⟩
expectation_value = np.vdot(statevector, transformed_state)

# Output the expectation value
print("Expectation value ⟨Ψ| Op |Ψ⟩:", expectation_value)


# In[3]:


# Tensor Representation
# In the tensor representation, the quantum state is represented as an n-dimensional tensor, where each dimension corresponds to a qubit. 
# Here, we use tensor contractions to apply the operator and compute the expectation value.

import numpy as np

# Define the number of qubits
n = 2  # Example with 2 qubits

# Example tensor state for 2 qubits, shape (2, 2)
# This state could be obtained from the output of a quantum circuit
# Representing  |Ψ+> Bell state as a tensor
tensor_state = np.array([[0.707, 0], [0, 0.707]])

# Define an example 2-qubit operator (Z ⊗ Z as above)
Z = np.array([[1, 0], [0, -1]])
Op = np.kron(Z, Z).reshape(2, 2, 2, 2)  # Reshape to apply as tensor

# Step 1: Apply the operator to the tensor state
# Contract over the corresponding qubits
transformed_tensor = np.tensordot(Op, tensor_state, axes=([2, 3], [0, 1]))

# Step 2: Flatten tensors for inner product calculation
# (alternatively, one can calculate directly using multi-index summation)
expectation_value = np.vdot(tensor_state.flatten(), transformed_tensor.flatten())

# Output the expectation value
print("Expectation value ⟨Ψ| Op |Ψ⟩:", expectation_value)

