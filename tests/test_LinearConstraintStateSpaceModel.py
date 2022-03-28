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

    def test_system4(self):
        system = LinearConstraintStateSpaceModel(
            A=np.array([
                [0.79898356, 0.93832862, 0.30943309, 0.9329607, 0.52567468],
                [0.43480883, 0.77582657, 0.71322907, 0.56725567, 0.02921253],
                [0.07447839, 0.74912897, 0.12599992, 0.27272414, 0.33534004],
                [0.30690427, 0.0917924, 0.29935492, 0.93765648, 0.56331568],
                [0.74560995, 0.43528731, 0.32654366, 0.00334108, 0.83872725]
            ]),
            B=np.array([
                [0.53678534, 0.93797791, 0.37248788],
                [0.50745662, 0.23616261, 0.58877284],
                [0.97512747, 0.32282102, 0.7786647],
                [0.54521594, 0.63116388, 0.27805539],
                [0.99799447, 0.72076249, 0.06001138]
            ]),
            G=np.array([
                [0.472, 0.19, 0.293, 0.351, 0.396],
                [0.627, 0.586, 0.134, 0.558, 0.203],
                [0.298, 0.282, 0.66, 0.604, 0.067],
                [10.878, 8.182, 8.036, 11.5, 5.261],
                [4.601, 3.32, 5.926, 6.19, 2.256]
            ]),
            F=np.array([
                [0.42111223, 0.08285465, 0.31751467, 0.50741614, 0.78448954, 0.39717398],
                [0.1612954, 0.89717344, 0.7428686, 0.14443119, 0.6704288, 0.03088425],
                [0.03849139, 0.73873068, 0.79369482, 0.13195193, 0.15142974, 0.61221381],
                [0.07526989, 0.12390503, 0.62527005, 0.36274025, 0.99327459, 0.25669884],
                [0.82678452, 0.27598196, 0.43322599, 0.59732486, 0.80116294, 0.01101946]
            ]),
            init_state=np.array([0.56375954, 0.18230589, 0.46795963, 0.79780289, 0.0667068])
        )
        system.ode_gain_solve(time_space=np.linspace(0, 20, int(2E3)))
        system.plot_states()

    def test_random_system(self):
        n = 4  # size of state vector
        u = 3  # size of controller vector
        k = n  # number of constraints
        A = np.random.rand(n, n)
        B = np.random.rand(n, u)
        F = np.random.rand(n, 6)
        state = np.random.rand(n)

        # Generate random row space basis
        row_space = 10 * np.round(np.random.random_sample() * np.random.rand(np.random.choice(np.arange(1, k)), n), 4)

        # Make sure rows are independent
        assert matrix_rank(row_space) == len(row_space), f"row_space vector:\n{row_space}\nnot independent, try again"

        part = np.asarray([
            np.sum(np.asarray([np.random.randint(0, 10) * vector for vector in row_space]), axis=0)
            for _ in range(k - len(row_space))
        ])  # Generate random linear combination of row space of length n - len(row_space)

        G = np.block([[row_space], [part]])  # Form Constraint matrix from row space
        R, _, _, N = subspaces_from_svd(G)

        system = LinearConstraintStateSpaceModel(A=A, B=B, G=G, F=F, init_state=state)
        system.ode_gain_solve(time_space=np.linspace(0, 20, int(2E3)))
        system.plot_states()

        print(f"A:\n{A}\n")
        print(f"B:\n{B}\n")
        print(f"G:\n{G}\n")
        print(f"F:\n{F}\n")
        print(f"initial state:\n{state}\n")
