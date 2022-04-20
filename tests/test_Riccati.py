from unittest import TestCase

from RiccatiEquation import RiccatiEquation
from ric_systems import *
from OrthogonalDecomposition import is_positive_definite


class TestRiccatiEquation(TestCase):

    def test_sys1(self):
        rand_sys = ric_random_system(n_size=8, u_size=5, k_size=4)  # Generates a random Constraint System

        riccati_sys = RiccatiEquation(rand_sys)  # Sets up the Riccati equations
        _, _, S_nn, phi = riccati_sys.solve()  # Solve the equations to get the Matrices coefficients

        print(f"{S_nn=}")
        print(f"{phi=}")

        # Should pass so long as lqr is solved
        assert is_positive_definite(S_nn), "S matrix not positive definite"

    def test_system7(self):
        system = ric_constraint_system7
        system.ode_gain_solve(time_space=np.linspace(0, 5, int(2E3)))
        system.plot_output()
