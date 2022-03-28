import numpy as np
from unittest import TestCase

from numpy.linalg import matrix_rank

from LinearConstraintStateSpaceModel import LinearConstraintStateSpaceModel
from OrthogonalDecomposition import subspaces_from_svd


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
        print(system.N)
        system.ode_gain_solve(time_space=np.linspace(0, 20, int(2E3)))
        system.plot_states()

    def test_random_system(self):
        n = 4
        u = 3
        A = np.random.randn(n, n)
        B = np.random.randn(n, u)
        F = np.random.randn(n, 6)
        state = np.random.randn(n)

        # Generate random row space basis
        row_space = 10 * np.random.random_sample() * np.random.randn(np.random.choice(np.arange(1, n)), n)
        assert matrix_rank(row_space) == len(row_space)  # Make sure rows are independent

        part = np.asarray([
            np.sum(np.asarray([np.random.random_sample() * vector for vector in row_space]), axis=0)
            for _ in range(n - len(row_space))
        ])  # Generate random linear combination of row space of length n - len(row_space)

        G = np.block([[row_space], [part]])  # Form Constraint matrix from row space
        R, _, _, N = subspaces_from_svd(G)
        print(f"G:\n{G}\n")
        print(f"{N}\n")
        print(f"{R}\n")

        system = LinearConstraintStateSpaceModel(A=A, B=B, G=G, F=F, init_state=state)
        system.ode_gain_solve(time_space=np.linspace(0, 10, int(2E3)))
        system.plot_states()

        print(f"A:\n{A}\n")
        print(f"B:\n{B}\n")
        print(f"F:\n{F}\n")
        print(f"initial state:\n{state}\n")
