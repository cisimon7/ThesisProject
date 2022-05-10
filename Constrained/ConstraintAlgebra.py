import numpy as np
from typing import Optional
from scipy.integrate import odeint
from Constrained.BaseConstraint import BaseConstraint


class ConstraintAlgebra(BaseConstraint):
    def __init__(self, A: np.ndarray, B: Optional[np.ndarray] = None, C: Optional[np.ndarray] = None,
                 D: Optional[np.ndarray] = None, G: Optional[np.ndarray] = None, F: Optional[np.ndarray] = None,
                 init_state: Optional[np.ndarray] = None):

        super().__init__(A, B, C, D, G, F, init_state)  # Runs the init method of the super class BaseConstraint

        self.zeta = self.R @ init_state

        self.gain_z = None  # gain for z state
        self.gain_zeta = np.linalg.pinv(self.N @ self.B) @ self.N @ self.A @ self.R.T  # From Equation 13

    def dynamics(self, state: np.ndarray, time: float, K_z: np.ndarray, k_0: np.ndarray) -> np.ndarray:
        """Returns a vector of the state derivative vector at a given state and controller gain"""

        (z, zeta, gain_zeta) = (state, self.zeta, self.gain_zeta)
        (A, B, N, R) = (self.A, self.B, self.N, self.R)

        self.assert_state_size(z)
        self.assert_zeta_size(zeta)
        self.assert_gain_size(K_z)

        # Line 13 from main paper
        control_zeta = - gain_zeta @ zeta

        if k_0 is not None:
            control_zeta += k_0

        # Substituting line 14 into line 9 from the main paper
        result = (N @ (A @ N.T - B @ K_z) @ z) + (N @ B @ control_zeta) + (N @ A @ R.T @ zeta)

        return result

    def gain_lqr(self, A=None, B=None, Q=None, R=None, set_gain=True):
        N = self.N

        _Q = np.eye(self.state_size - self.rank_G) if (Q is None) else Q
        _R = np.eye(self.control_size) if (R is None) else R

        _A = (N @ self.A @ N.T) if (A is None) else A
        _B = (N @ self.B) if (B is None) else B

        # TODO(Check for controllability of given matrix _A and _B)

        gain = super().gain_lqr(_A, _B, _Q, _R)
        if set_gain:
            self.gain_z = gain

        return gain

    def ode_gain(self, gain=None, control_const=None, init_state=None,
                 time_space: np.ndarray = np.linspace(0, 10, int(2E3)), verbose=False):

        self.time = time_space

        _init_state = self.init_state if (init_state is None) else init_state
        self.init_z_state = self.N @ _init_state
        self.zeta = self.R @ _init_state

        # uses identity matrix for Q and R to get an initial lqr gain if gain is not specified
        _gain = self.gain_lqr() if (gain is None) else gain
        self.gain_z = _gain

        result = odeint(self.dynamics, self.init_z_state, self.time, args=(_gain, control_const), printmessg=verbose)

        z_states = np.asarray(result).transpose()
        self.z_states = z_states
        d_z_states = np.asarray(
            [self.dynamics(state, time=t, K_z=_gain, k_0=control_const) for (t, state) in
             zip(self.time, result)]
        ).transpose()

        cons_zeta = self.R.T @ self.zeta * 0  # TODO(Adding zeta makes state not tend to zero)

        # Calibrating to remove constant zeta gain
        self.states = self.N.T @ z_states + np.asarray([cons_zeta for _ in range(z_states.shape[1])]).T
        self.d_states = self.N.T @ d_z_states

        self.controller = - _gain @ self.N @ self.states
        self.output()

        return self.states, self.d_states
