import numpy as np
from unittest import TestCase
from TimeInVaryingAffineSystem import TimeInVaryingAffineSystem
from Unconstrained.LTIWithConstantTerm import LTIWithConstantTerm
from Unconstrained.system import random_lti_system_with_const_term


class TestTimeInVaryingAffineSystem(TestCase):
    def setUp(self) -> None:
        pass

    def test_system(self):
        system = LTIWithConstantTerm(
            A=np.eye(4),
            B=np.eye(4),
            c=2 * np.ones(4).T
        )
        system.ode_solve(time_space=np.linspace(0, 10, int(2E3)))
        system.plot_states()
        # system.plot_controller()

    def test_random_system(self):
        system = random_lti_system_with_const_term(n_size=4, u_size=3)
        system.ode_solve(time_space=np.linspace(0, 10, int(2E3)))
        system.plot_states()
        # system.plot_controller()
