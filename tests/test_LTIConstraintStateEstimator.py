from unittest import TestCase
import numpy as np

from LTIConstraintStateEstimator import LTIConstraintStateEstimator
from tests.test_systems import *


class TestLTIConstraintStateEstimator(TestCase):

    def test_state_dot(self):
        system = LTIConstraintStateEstimator(constraint_system2)
        dot = system.state_dot_hat(x_state=system.system.init_state, time=0,
                                   L_gain=np.eye(system.system.state_size), K_gain=system.system.gain_lqr())
        print(dot)

    def test_system2(self) -> None:
        system = LTIConstraintStateEstimator(constraint_system2)
        system.estimate(time_space=np.linspace(0, 20, int(2E3)))

        system.plot_states()
        system.plot_controller()
        system.plot_output()

    def test_system4(self):
        system = LTIConstraintStateEstimator(constraint_system4)
        system.estimate(time_space=np.linspace(0, 15, int(2E3)))
        system.plot_states()
        system.plot_controller()
        system.plot_output()
