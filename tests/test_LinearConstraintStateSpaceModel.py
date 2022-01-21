import numpy as np
from unittest import TestCase

from LinearConstraintStateSpaceModel import LinearConstraintStateSpaceModel


class TestLinearConstraintStateSpaceModel(TestCase):

    def setUp(self) -> None:
        # Good A and B pairs gotten from random generation
        self.A1 = np.array([[-1.01416031, -0.85202817, -0.23718763],
                            [-0.24703406, -1.20508211, 0.47151586],
                            [0.37052781, 1.13780232, -0.05080365]])

        self.B1 = np.array([[1.62100904, 1.11382971, 1.68311231, 0.90885391],
                            [-0.51764834, -1.15764292, -0.53954765, -0.13238155],
                            [-0.60691388, -0.50319479, 0.79373825, 1.07906922]])

        self.A2 = np.array([[1.96692622e+00, -7.63863478e-01, 1.98132371e+00],
                            [-8.38705840e-01, -8.92497879e-01, 1.71463468e-04],
                            [1.09646714e+00, 2.14742868e+00, 1.69597482e+00]])

        self.B2 = np.array([[1.08525237, 0.56397236, 0.3333663, -0.31324784],
                            [1.40949541, -0.71419962, 0.84513168, -0.01412536],
                            [-0.57914272, 0.35263826, 0.60486569, 1.91493988]])

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
        system = LinearConstraintStateSpaceModel(
            A=self.A1,
            B=self.B1,
            G=np.array([
                [1, 2, 3],
                [1, 1, 2],
                [1, 2, 3]
            ]),
            F=np.eye(3, 6),
            init_state=np.array([3, 5, 10])
        )
        system.ode_gain_solve(time_space=np.linspace(0, 20, int(2E3)))
        system.plot_states()

    def test_system4(self):
        system = LinearConstraintStateSpaceModel(
            A=self.A2,
            B=self.B2,
            G=np.array([
                [1, 2, 3],
                [1, 1, 2],
                [1, 2, 3]
            ]),
            F=np.eye(3, 6),
            init_state=2 * np.random.randn(3)
        )
        system.ode_gain_solve(time_space=np.linspace(0, 60, int(2E3)))
        system.plot_states()
        system.plot_controller()
        system.plot_output()

    def test_rand_system(self):
        A = np.random.randn(3, 3)
        B = np.random.randn(3, 4)
        system = LinearConstraintStateSpaceModel(
            A=A,
            B=B,
            G=np.array([
                [1, 2, 3],
                [1, 1, 2],
                [1, 2, 3]
            ]),
            F=np.eye(3, 6),
            init_state=np.random.randn(3)
        )
        system.ode_gain_solve(time_space=np.linspace(0, 20, int(2E3)))
        system.plot_states()
        print(A)
        print(B)
