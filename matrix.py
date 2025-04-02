import numpy as np
from scipy.linalg import inv, det, eig, orth

# Define matrices
A = np.array([[2, 3], [1, 4]])
B = np.array([[5, 2], [3, 1]])

# Matrix operations
addition = A + B
subtraction = A - B
multiplication = A @ B  # Matrix multiplication
elementwise_multiplication = A * B
determinant_A = det(A)
inverse_A = inv(A) if determinant_A != 0 else "Singular Matrix (No Inverse)"

eigenvalues, eigenvectors = eig(A)

# Check orthogonality of eigenvectors (dot product should be close to identity matrix)
orthogonality_check = np.allclose(eigenvectors.T @ eigenvectors, np.eye(eigenvectors.shape[1]))

# Define a set of vectors
V = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Compute basis (orthonormal basis using Gram-Schmidt process)
basis = orth(V)

# Print results
print("Addition:\n", addition)
print("\n")
print("Subtraction:\n", subtraction)
print("\n")
print("Matrix Multiplication:\n", multiplication)
print("\n")
print("Element-wise Multiplication:\n", elementwise_multiplication)
print("\n")
print("Determinant of A:", determinant_A)
print("\n")
print("Inverse of A:\n", inverse_A)
print("\n")
print("Eigenvalues:\n", eigenvalues)
print("\n")
print("Eigenvectors:\n", eigenvectors)
print("\n")
print("Are Eigenvectors Orthogonal?:", orthogonality_check)
print("\n")
print("Basis of Vector Space:\n", basis)
print("\n")
