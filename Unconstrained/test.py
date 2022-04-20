import numpy as np
from unittest import TestCase
from TimeInVaryingAffineSystem import TimeInVaryingAffineSystem


class TestTimeInVaryingAffineSystem(TestCase):
    def setUp(self) -> None:
        pass

    def test_system(self):
        system = TimeInVaryingAffineSystem(
            A=np.eye(4),
            B=np.eye(4),
            C=np.ones(4).T
        )
        system.ode_solve(time_space=np.linspace(0, 10, int(2E3)))
        system.plot_states()
        # system.plot_controller()
