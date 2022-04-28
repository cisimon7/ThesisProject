from unittest import TestCase
from ric_systems import *


class TestRiccatiEquation(TestCase):

    def test_sys1(self):
        rand_sys = ric_random_system(n_size=8, u_size=5, k_size=4)  # Generates a random Constraint System
        sys = ConstraintRiccatiSystem(rand_sys)
        sys.ode_solve()
        sys.plot_output()

    def test_system7(self):
        system = ric_constraint_system5
        system.ode_solve(time_space=np.linspace(0, 10, int(2E3)))
        system.pplot_states()
