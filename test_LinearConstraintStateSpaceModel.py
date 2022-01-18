import numpy as np
from unittest import TestCase

from LinearConstraintStateSpaceModel import LinearConstraintStateSpaceModel


class TestLinearConstraintStateSpaceModel(TestCase):

    def test_system1(self):
        system = LinearConstraintStateSpaceModel(
            A=np.eye(4),
            B=np.array([
                [0, 1 / 67, 0, 1 / 67],
                [0, -(4 / 67), 0, -(4 / 67)],
                [0, 7 / 67, 0, 7 / 67],
                [0, 1 / 67, 0, 1 / 67]
            ]),
            G=np.array([[1, 4, 2, 1], [3, 3, 1, 2], [0, 1, 0, 4]]),
            # G=np.array([[1, 0, -1, 1], [0, 1, 1, 1], [1, 1, 0, 2], [2, 3, 1, 5]]),
            F=np.eye(4, 6),
            init_state=10 * np.array([1, 1, 1, 1])
        )
        system.ode_gain_solve(time_space=np.linspace(0, 10, int(2E3)))
        system.plot_states()

    def test_system2(self):
        system = LinearConstraintStateSpaceModel(
            A=np.eye(4),
            B=np.eye(4),
            G=np.array([
                [1, 4, 2, 1],
                [3, 3, 1, 2],
                [0, 1, 0, 4]
            ]),
            F=np.eye(4, 6),
            init_state=10 * np.array([1, 1, 1, 1])
        )
        system.ode_gain_solve(time_space=np.linspace(0, 10, int(2E3)))
        system.plot_states()

    def test_system3(self):
        pass
