from unittest import TestCase
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


"""## Proves"""

# G = 100 * np.random.randn(5, 5)
# x_ = 100 * np.random.randn(5)
# R, _, _, N = svd_4subspaces(G)

# print(projection_matrix(R).round(4) == (R @ R.T).round(4))

# projection_matrix(N).round(4) == (N@N.T).round(4)

# print(x_.round(4) == (R @ R.T @ x_).round(4))  # + (N@N.T@x).round(4)
