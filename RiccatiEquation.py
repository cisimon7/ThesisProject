import numpy as np
from control import lqr
from LinearConstraintStateSpaceModel import LinearConstraintStateSpaceModel


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

        K, S_nn, _ = lqr(A_nn, B_n, Q, R)
        phi = - np.linalg.inv(2 * A_nn.T - S_nn @ B_n @ np.linalg.inv(R) @ B_n.T) @ S_nn @ A_nr @ zeta

        K_z = np.linalg.inv(R) @ B_n.T @ S_nn
        K_phi = - np.linalg.inv(R) @ B_n.T @ phi

        return K_z, K_phi, S_nn, phi
