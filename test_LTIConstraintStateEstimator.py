from unittest import TestCase
import numpy as np

from LTIConstraintStateEstimator import LTIConstraintStateEstimator
from LinearConstraintStateSpaceModel import LinearConstraintStateSpaceModel


class TestLTIConstraintStateEstimator(TestCase):

    def setUp(self) -> None:
        A1 = np.array([[-1.01416031, -0.85202817, -0.23718763],
                       [-0.24703406, -1.20508211, 0.47151586],
                       [0.37052781, 1.13780232, -0.05080365]])

        B1 = np.array([[1.62100904, 1.11382971, 1.68311231, 0.90885391],
                       [-0.51764834, -1.15764292, -0.53954765, -0.13238155],
                       [-0.60691388, -0.50319479, 0.79373825, 1.07906922]])

        self.system = LinearConstraintStateSpaceModel(
            A=A1,
            B=B1,
            G=np.array([
                [1, 2, 3],
                [1, 1, 2],
                [1, 2, 3]
            ]),
            F=np.eye(3, 6),
            init_state=np.array([3, 5, 10])
        )
        self.estimator = LTIConstraintStateEstimator(self.system)

    def test_z_dot_hat_gain(self):
        self.fail()

    def test_estimator(self):
        self.estimator.estimate()
