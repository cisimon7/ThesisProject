import numpy as np
import sympy as sp
from scipy.linalg import orth, svd


def test_product_with_transpose(mat: np.ndarray) -> bool:
    (x, y) = mat.shape
    return np.array_equal(
        mat @ np.transpose(mat),
        np.eye(x)
    )


def transpose_inverse(mat: np.ndarray) -> bool:
    return np.array_equal(
        np.transpose(mat),
        np.linalg.inv(mat)
    )


def is_orthonormal(mat: np.ndarray) -> bool:
    """
    Implementation also returns true if matrix is only Semi-Orthogonal
    Implemented this way to take care of the floating point precision
    """
    Q = np.asarray(mat)
    (x, y) = Q.shape

    if np.sum(
            np.square(np.eye(x).flatten() - (Q @ Q.T).flatten())
    ) < (1e-4): return True

    if np.sum(
            np.square(np.eye(y).flatten() - (Q @ Q.T).flatten())
    ) < (1e-4): return True

    return False


def rnd_orth(size: int) -> np.ndarray:
    """
    Generate Random Orthonormal Square matrices
    """
    a = np.random.randn(size, size)
    return orth(a)  # Uses SVD


all([is_orthonormal(rnd_orth(4)) for _ in range(10_000)])

"""## Rank and Subspaces"""


def matrix_orth(mat: np.ndarray) -> np.ndarray:
    """Returns Orthonormal basis for the matrix"""
    return orth(mat)


def matrix_rref(mat: np.ndarray) -> np.ndarray:
    """Return Row Reduced Echelon Form of the matrix"""
    ref, _ = sp.Matrix(mat).rref()
    rref = np.array(ref).astype(np.float64).round(4)
    return rref


def matrix_rank(mat: np.ndarray) -> int:
    """Returns Rank of the matrix"""
    return np.linalg.matrix_rank(mat)


def col_space_basis(mat: np.ndarray) -> np.ndarray:
    """Returns Column Space (not orthognal) for the matrix from its columns"""
    rank = matrix_rank(mat)
    return mat[:, :rank - 1]


def row_space_basis(mat: np.ndarray) -> np.ndarray:
    """Returns Row space (not orthogonal) for the matrix from its rows"""
    return col_space_basis(mat.T).T


def svd_4subspaces(mat: np.ndarray):
    """Returns orthognal basis for the four fundamental subspaces"""
    rank = matrix_rank(mat)
    U, s, V = svd(mat, full_matrices=True)

    # Row       # Column    # Left-Null # Null
    return U[:, :rank], V[:rank, :], U[:, :rank], V[rank:, :]


# TEST: SVD decomposition should be able to return the 4 sub spaces, and all orthornomal
a = np.random.randn(5, 7)
row, col, left_null, null = svd_4subspaces(a)
all([is_orthonormal(basis) for basis in [row, col, left_null, null]])


def projection_vector(span, vector):
    u = span
    v = vector
    return (u.v) / (u.u) * u


def projection_matrix(mat: np.ndarray):
    U = mat
    return U @ np.linalg.inv(U.T @ U) @ U.T


def intersection_basis(subspace1: np.ndarray, subspace2: np.ndarray) -> np.ndarray:
    U = subspace1
    V = subspace2
    Pu = projection_matrix(U)
    Pv = projection_matrix(V)
    M = Pu @ Pv
    w, v = np.linalg.eig(M)
    return v


def rnd_vec_from_basis(basis: np.ndarray) -> np.ndarray:
    (x, y) = basis.shape
    vec = np.random.randint(0, 10, x)
    return (basis @ vec)


def is_vec_in_span(span, vec) -> bool:
    (x, y) = span.shape
    ref, _ = sp.Matrix(np.hstack([
        np.asarray(span),
        np.c_[np.asarray(vec)]
    ])).rref()
    rrech = np.array(ref).astype(np.float64).round(4)

    mat, vec = rrech[:, :y], rrech[:, -1]

    if np.array_equal(
            rrech[:, :y],
            np.eye(x)
    ): return True  # Case of Single Solution

    # if any([
    #       all(np.where(row==0, True, False)) and (elem == 0)
    #       for (row, elem) in zip(mat,vec.ravel())
    # ]): return True # Case of Infinite Solution

    if any([
        all(np.where(row == 0, True, False)) and (elem != 0)
        for (row, elem) in zip(mat, vec.ravel())
    ]): return False  # Case of No Solution

    return True


# Testing with Random Orthornomal Basis Matrices
# a = rnd_orth(5)
# b = rnd_orth(5)
# intsc = intersection_basis(a, b)
#
# all([is_vec_in_span(span, vector)
#      for span in [a, b]
#      for vector in [rnd_vec_from_basis(intsc) for _ in range(1_000)]
#      ])

"""## Proves"""

# G = 100 * np.random.randn(5, 5)
# x_ = 100 * np.random.randn(5)
# R, _, _, N = svd_4subspaces(G)

# print(projection_matrix(R).round(4) == (R @ R.T).round(4))

# projection_matrix(N).round(4) == (N@N.T).round(4)

# print(x_.round(4) == (R @ R.T @ x_).round(4))  # + (N@N.T@x).round(4)
