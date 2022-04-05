from unittest import TestCase
from RiccatiEquation import RiccatiEquation
from tests.test_systems import constraint_system2, random_system
from OrthogonalDecomposition import is_positive_definite, is_symmetric


class TestRiccatiEquation(TestCase):

    def test_sys1(self):
        rand_sys = random_system(n_size=8, u_size=5, k_size=4)  # Generates a random Constraint System

        riccati_sys = RiccatiEquation(constraint_system2)  # Sets up the Riccati equations
        S_nn, phi = riccati_sys.solve()  # Solve the equations to get the Matrices coefficients

        print(f"{S_nn=}")
        print(f"{phi=}")

        # Should pass so long as lqr is solved
        assert is_positive_definite(S_nn), "S matrix not positive definite"
