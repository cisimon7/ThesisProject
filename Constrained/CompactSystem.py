import numpy as np
from control import lqr
from typing import Optional
from scipy.integrate import odeint
from Constrained.BaseConstraint import BaseConstraint


class CompactSystem(BaseConstraint):
    def __init__(self, A: np.ndarray, B: Optional[np.ndarray] = None, C: Optional[np.ndarray] = None,
                 D: Optional[np.ndarray] = None, G: Optional[np.ndarray] = None, F: Optional[np.ndarray] = None,
                 init_state: Optional[np.ndarray] = None):
        super().__init__(A, B, C, D, G, F, init_state)
        (self.r, self.l) = (self.rank_G, self.state_size - self.rank_G)
        (r, l) = (self.r, self.l)

        # Denotation to avoid repetition
        self.A_nn = self.N @ A @ self.N.T
        self.A_nr = self.N @ A @ self.R.T
        self.B_n = self.N @ B

        # New A matrix
        self.A_z_zeta = np.block([
            [self.A_nn, self.A_nr],
            [np.zeros((r, l)), 10e-10 + np.zeros((r, r))]
        ])

        # New B matrix
        self.B_z_zeta = np.block([
            [self.B_n],
            [np.zeros((r, self.control_size))]
        ])

        self.init_z_state = None
        self.zeta = self.R @ init_state

    def dynamics(self, state: np.ndarray, time: float, K_z: np.ndarray, k_0: np.ndarray) -> np.ndarray:
        A, B = self.A_z_zeta, self.B_z_zeta

        z_dot = (A @ state) + (B @ ((-K_z @ state) - k_0))
        return z_dot

    def riccati_solve(self, Q=None, R=None):
        A, B = self.A_z_zeta, self.B_z_zeta
        (r, l) = (self.r, self.l)

        Q = np.eye(r + l) if Q is None else Q
        Q_nn = (self.N @ Q @ self.N.T).round(4)
        Q_nr = (self.N @ Q @ self.R.T).round(4)
        Q_rr = (self.R @ Q @ self.R.T).round(4)

        Q = np.block([
            [Q_nn, Q_nr],
            [Q_nr.T, Q_rr]
        ]) if Q is None else Q
        R = np.eye(B.shape[1]) if R is None else R

        iR = np.linalg.pinv(R)

        K, S_z_zeta, _ = lqr(A, B, Q, R)

        k_z = iR @ B.T @ S_z_zeta
        k_0 = np.zeros((B.shape[1], 1)).reshape(-1, )

        return k_z, k_0

    def ode_solve(self, k_z=None, k_0=None, init_state=None, time_space=np.linspace(0, 10, int(2E3)), verbose=False):
        R, N = self.R, self.N
        r, l = self.r, self.l

        self.time = time_space

        _init_state = self.init_state if (init_state is None) else init_state
        self.init_z_state = self.N @ _init_state
        self.zeta = self.R @ _init_state

        init = np.r_[self.init_z_state, self.zeta]

        (k_z, k_0) = self.riccati_solve() if (k_z is None or k_0 is None) else (k_z, k_0)

        result = np.asarray(odeint(
            self.dynamics,
            init,
            self.time,
            args=(k_z, k_0),
            printmessg=verbose
        ))

        z_zeta = result.T
        d_z_zeta = np.asarray([
            self.dynamics(state, time=t, K_z=k_z, k_0=k_0)
            for (t, state) in zip(self.time, result)
        ]).transpose()

        self.z_states = z_zeta[:l, :]

        x = np.block([[N.T, R.T]]) @ z_zeta
        self.states = x

        cons_zeta = self.R.T @ self.zeta * 0  # TODO(Adding zeta makes state not tend to zero)

        self.states = self.N.T @ self.z_states + np.asarray([cons_zeta for _ in range(self.z_states.shape[1])]).T
        self.d_states = self.N.T @ d_z_zeta[:l, :]

        self.controller = np.asarray([- (k_z @ z_ze) for z_ze in z_zeta.T]).T
        self.output()

        return self.states, self.d_states
