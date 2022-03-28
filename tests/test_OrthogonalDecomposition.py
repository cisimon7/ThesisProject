from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal

from OrthogonalDecomposition import *


class TestOrthogonalDecomposition(TestCase):

    def test_generate_and_test_random_orthonormal_matrices(self):
        # 10_000 orthonormal matrices randomly generated and tested
        self.assertTrue(all([is_orthonormal(rnd_orth(4)) for _ in range(10_000)]))

    def test_4_subspaces_are_orthonormal(self):
        # SVD decomposition should be able to return the 4 sub spaces, and are all orthonormal
        a = np.random.randn(5, 7)  # generate a random matrix
        row, col, left_null, null = subspaces_from_svd(a)  # retrieve the 4 fundamental subspaces
        self.assertTrue(
            all([is_orthonormal(basis) for basis in [row, col, left_null, null]])  # test if they are all orthonormal
        )

    def test_dimensions_of_subspaces(self):
        pass

    def test_intersection_of_subspaces(self):
        a = rnd_orth(5)  # generate random orthonormal matrix
        b = rnd_orth(5)  # another random orthonormal matrix
        intersection = intersection_basis(a, b)  # find intersection of the two basis

        self.assertTrue(all([
            # test if generated vector from intersection basis is in span of basis of a subspace
            is_vec_in_span(basis, vector)
            # initial orthonormal matrices as basis
            for basis in [a, b]
            # 1_000 random linear combination of matrix columns
            for vector in [rnd_vec_from_basis(intersection) for _ in range(1_000)]
        ]))

    def test_subspaces_are_orthogonal(self):
        G = np.array([[1, 4, 2, 1], [3, 3, 1, 2], [0, 1, 0, 4]])
        R, C, LN, N = subspaces_from_svd(G)  # the fundamental subspaces gotten from svd are orthogonal basis

        self.assertTrue(is_orthonormal(R))  # Row Space
        self.assertTrue(is_orthonormal(C))  # Column Space
        # self.assertTrue(is_orthonormal(LN))  # Left Null Space is empty in this case
        self.assertTrue(is_orthonormal(N))  # Null Space

    def test_pseudo_inverse_of_orthonormal_matrices_equals_transpose(self):
        G = np.array([[1, 4, 2, 1], [3, 3, 1, 2], [0, 1, 0, 4]])
        R, C, LN, N = subspaces_from_svd(G)  # from previous test, the fundamental subspaces from svd are orthogonal

        assert_almost_equal(
            np.linalg.pinv(R),
            R.T
        )

        assert_almost_equal(
            np.linalg.pinv(N),
            N.T
        )

        assert_almost_equal(
            np.linalg.pinv(C),
            C.T
        )

    def test_projection_matrix_of_orthonormal_basis_equals_product_matrix_and_its_transpose(self):
        G = np.array([[1, 4, 2, 1], [3, 3, 1, 2], [0, 1, 0, 4]])
        R, _, _, N = subspaces_from_svd(G)  # The subspaces are orthonormal matrices

        # Testing projection into row space of RowSpace basis
        assert_almost_equal(
            R.T @ np.linalg.inv(R @ R.T) @ R,
            (R.T @ R)
        )

        # Testing projection to the null space
        assert_almost_equal(
            np.eye(4) - R.T @ np.linalg.inv(R @ R.T) @ R,
            (N.T @ N)
        )

    def test_decomposing_vector_into_row_and_null_space(self):
        G = np.array([[1, 4, 2, 1], [3, 3, 1, 2], [0, 1, 0, 4]])
        x_ = 10 * np.random.randn(4)
        R, _, _, N = subspaces_from_svd(G)

        z_ = N @ x_
        zeta = R @ x_

        assert_almost_equal(
            x_,
            (N.T @ z_) + (R.T @ zeta)
        )

    def test_svd(self):
        G = np.array([
            [5.55, 1.71, 3.145, 2.943],
            [44.4, 13.68, 25.16, 23.544],
            [5.55, 1.71, 3.145, 2.943],
            [16.65, 5.13, 9.435, 8.829]
        ])

        R, _, _, N = subspaces_from_svd(G)
        print(R)
        print(N)
