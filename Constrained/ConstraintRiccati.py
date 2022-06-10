from typing import Optional
import numpy as np
from control import lqr
from scipy.integrate import odeint
from Constrained.BaseConstraint import BaseConstraint


class ConstraintRiccatiSystem(BaseConstraint):
    def __init__(self, A: np.ndarray, B: Optional[np.ndarray] = None, C: Optional[np.ndarray] = None,
                 D: Optional[np.ndarray] = None, G: Optional[np.ndarray] = None, F: Optional[np.ndarray] = None, g=None,
                 init_state: Optional[np.ndarray] = None, x_desired: Optional[np.ndarray] = None,
                 dx_desired: Optional[np.ndarray] = None):
        # Runs the init method of the super class BaseConstraint
        super().__init__(A, B, C, D, G, F, g, init_state, x_desired, dx_desired)

        # Denotation to avoid repetition
        self.A_nn = self.N @ A @ self.N.T
        self.A_nr = self.N @ A @ self.R.T
        self.B_n = self.N @ B

        self.alpha = 1
        self.ext_u0 = True
        self.U_0 = np.linalg.pinv(self.B_n) @ self.A_nr @ self.zeta

    def dynamics(self, state: np.ndarray, time: float, K_z: np.ndarray, k_0: np.ndarray) -> np.ndarray:
        """Returns a vector of the state derivative vector at a given state and controller gain"""

        (z, zeta) = (state - (self.N @ self.x_desired), self.zeta - (self.R @ self.x_desired))
        (A_nn, A_nr, B_n) = (self.A_nn, self.A_nr, self.B_n)

        self.assert_state_size(z)
        self.assert_zeta_size(zeta)
        self.assert_gain_size(K_z)

        z_dot = A_nn @ z + (B_n @ ((-K_z @ z) - k_0)) + (A_nr @ zeta) + (self.N @ self.dx_desired)
        return z_dot

    def riccati_solve(self, Qz=None, Ru=None):
        """
        J  = z.T Q z  +  u.T R u
        J* = x.T Sxx x  + sx.T x
        """
        (A_nn, A_nr, B_n) = (self.A_nn, self.A_nr, self.B_n)
        zeta = self.zeta

        Q = np.eye(A_nn.shape[0]) if Qz is None else Qz
        R = np.eye(B_n.shape[1]) if Ru is None else Ru

        if not self.ext_u0:
            R = self.alpha * R

        iR = np.linalg.pinv(R)

        _, S_nn, _ = lqr(A_nn, B_n, Q, R)
        phi = - np.linalg.pinv(A_nn.T - S_nn @ B_n @ iR @ B_n.T) @ S_nn @ A_nr @ zeta

        k_z = iR @ B_n.T @ S_nn

        if self.ext_u0:
            k_0 = + self.U_0
        else:
            k_0 = iR @ B_n.T @ phi

        return k_z, k_0

    def ode_solve(self, k_z=None, k_0=None, init_state=None, time_space: np.ndarray = np.linspace(0, 10, int(2E3)),
                  verbose=False):
        self.time = time_space

        _init_state = self.init_state.flatten() if (init_state is None) else init_state
        self.init_z_state = self.N @ _init_state
        self.zeta = self.R @ _init_state

        (k_z, k_0) = self.riccati_solve() if (k_z is None or k_0 is None) else (k_z, k_0)

        result = np.asarray(odeint(
            self.dynamics,
            self.init_z_state,
            self.time,
            args=(k_z, k_0),
            printmessg=verbose
        ))

        print(f"{(self.R @ self.init_state).flatten() - (self.R @ self.x_desired).flatten()}")

        z_states = result.T
        dz_states = np.asarray([
            self.dynamics(state, time=t, K_z=k_z, k_0=k_0)
            for (t, state) in zip(self.time, result)
        ]).transpose()
        self.dz_states = dz_states

        # cons_zeta = self.R.T @ self.zeta * 0  # TODO(Adding zeta makes state not tend to zero)
        # self.states = self.N.T @ z_states + np.asarray([cons_zeta for _ in range(z_states.shape[1])]).T

        cons_zeta = self.R.T @ self.zeta

        self.z_states = z_states
        self.states = self.N.T @ z_states + np.asarray([cons_zeta for _ in range(z_states.shape[1])]).T
        self.d_states = self.N.T @ dz_states

        self.controller = np.asarray([- (k_z @ z) for z in z_states.T]).T
        self.output()

        return self.states, self.d_states
