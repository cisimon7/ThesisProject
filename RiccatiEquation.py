import numpy as np
from control import lqr
from Constrained.LinearConstraintStateSpaceModel import LinearConstraintStateSpaceModel


class RiccatiEquation:
    def __init__(self, system: LinearConstraintStateSpaceModel):
        (N, R) = (system.N, system.R)
        (Ac, Bc) = (system.A, system.B)

        self.A_nn = N @ Ac @ N.T
        self.A_nr = N @ Ac @ R.T

        self.B_n = N @ Bc
        self.zeta = system.zeta

    def solve(self, Q=None, R=None):
        (A_nn, A_nr, B_n) = (self.A_nn, self.A_nr, self.B_n)
        zeta = self.zeta

        Q = np.eye(A_nn.shape[0]) if Q is None else Q
        R = np.eye(B_n.shape[1]) if R is None else R

        iR = np.linalg.pinv(R)

        _, S_nn, _ = lqr(A_nn, B_n, Q, R)
        phi = - np.linalg.pinv(A_nn.T - S_nn @ B_n @ iR @ B_n.T) @ S_nn @ A_nr @ zeta

        K_z = iR @ B_n.T @ S_nn
        const = - iR @ B_n.T @ phi

        return K_z, const, S_nn, phi

    def solve2(self, Q=None, R=None):
        (A_nn, A_nr, B_n) = (self.A_nn, self.A_nr, self.B_n)
        zeta = self.zeta

        Q = np.eye(A_nn.shape[0]) if Q is None else Q
        R = np.eye(B_n.shape[1]) if R is None else R

        iR = np.linalg.pinv(R)

        _, S, _ = lqr(A_nn, B_n, Q, R)
        s_z = - np.linalg.pinv((-0.5 * S @ B_n @ iR @ B_n.T) + (0.5 * A_nn)) @ (S @ A_nr @ zeta)

        K_z = iR @ B_n.T @ S
        const = np.linalg.pinv(B_n) @ A_nr @ zeta # - iR @ B_n.T @ s_z

        return K_z, const, S, s_z
