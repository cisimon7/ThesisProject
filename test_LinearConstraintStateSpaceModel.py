import numpy as np
from unittest import TestCase

from LinearConstraintStateSpaceModel import LinearConstraintStateSpaceModel


class TestLinearConstraintStateSpaceModel(TestCase):

    def setUp(self) -> None:
        self.rnd_system = LinearConstraintStateSpaceModel(
            A=np.eye(4),
            B=np.array([
                [0, 1 / 67, 0, 1 / 67],
                [0, -(4 / 67), 0, -(4 / 67)],
                [0, 7 / 67, 0, 7 / 67],
                [0, 1 / 67, 0, 1 / 67]
            ]),
            # G=np.array([[1, 4, 2, 1], [3, 3, 1, 2], [0, 1, 0, 4]]),
            G=np.array([[1, 0, -1, 1], [0, 1, 1, 1], [1, 1, 0, 2], [2, 3, 1, 5]]),
            F=np.eye(4, 6)
        )

    def test_zdot_gain(self):
        self.skipTest("Not yet implemented")

    def test_ode_gain_solve(self):
        self.rnd_system.ode_gain_solve()
        self.assertTrue(True)
