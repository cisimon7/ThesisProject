from dataclasses import dataclass
from typing import Optional

import numpy as np
from control import lqe, lqr
from scipy.integrate import odeint

from ConstraintAlgebra import ConstraintAlgebra
from RiccatiEquation import RiccatiEquation


@dataclass
class ZDotHatParams:
    """Class for storing parameters for the function z_dot_hat_gain to be passed to odeint"""
    gain: Optional[np.ndarray]


class LTIConstraintStateEstimator:

    def __init__(self, system: ConstraintAlgebra):
        self.system = system
        (N, R) = (system.N, system.R)  # NullSpace and RowSpace
        (A, B) = (system.A, system.B)  # (Ac, Bc)
        (C, D) = (system.C, system.D)  #
        (self.r, self.l) = (system.rank_G, system.state_size - system.rank_G)  # Dimension of R, Dimension of N
        (r, l) = self.r, self.l

        # Components of Equation 15

        print(f"{A.shape=}")

        self.A = np.block([
            [N @ A @ N.T, N @ A @ R.T],
            [np.zeros((r, l)), np.zeros((r, r))]
        ])

        self.B = np.block([
            [N @ B],
            [np.zeros((r, system.control_size))]
        ])

        self.C = C @ np.block([[N.T, R.T]])
        # TODO(Check for observability of C)

        self.D = D

        self.zeta = R @ self.system.init_state

    def state_dot_hat(self, x_state: np.ndarray, time, L_gain: np.ndarray, K_gain: np.ndarray,
                      control_constant: np.ndarray) -> np.ndarray:
        """Returns an estimate of the state vector derivative vector at a given state and controller gain"""

        (A, B, C, D) = (self.A, self.B, self.C, self.D)
        (N, R) = (self.system.N, self.system.R)

        p_sig = 0.1
        p_mu = 0
        process_noise = p_sig * np.random.randn(x_state.shape[0]) + p_mu

        z_hat = N @ x_state  # z state
        zeta_hat = R @ x_state  # zeta state
        state_hat = np.r_[z_hat, zeta_hat]

        # To be received from measurement, contains errors
        y_actual = (C @ state_hat)

        sigma = 1
        mu = 0
        measurement_noise = sigma * np.random.randn(y_actual.shape[0]) + mu

        y_actual = y_actual + (0 * measurement_noise)  # Adding noise to simulate sensor noise

        U_z = - K_gain @ z_hat
        if control_constant is not None:
            U_z += control_constant

        U_zeta = - self.system.gain_zeta @ self.system.zeta

        es_B = np.block([[B, L_gain]])
        es_input = np.r_[(U_z + U_zeta), y_actual]

        # Equation 16 re-arranged
        d_state_estimate = ((A - L_gain @ C) @ state_hat) + (es_B @ es_input) + (0 * process_noise)

        d_z_ = d_state_estimate[:self.l]
        d_zeta_ = d_state_estimate[self.l:]

        d_x = N.T @ d_z_

        return d_x

    def estimate(self, k_gain=None, L_gain=None, control_constant=None, time_space=None, verbose=False):
        # L_gain, _, _ = lqe(
        #     self.A,
        #     np.eye(self.A.shape[0]),
        #     self.C,
        #     np.eye(self.A.shape[0]),
        #     np.eye(self.C.shape[1])
        # ) if params.gain is None else params.gain  # Estimator Gain

        L_gain = self.system.gain_lqr(
            A=self.A.T,
            B=self.C.T,
            Q=np.eye(self.A.T.shape[0]),
            R=np.eye(self.C.T.shape[1])
        ) if L_gain is None else L_gain  # Estimator Gain

        k_gain = self.system.gain_lqr() if k_gain is None else k_gain  # Controller Gain
        x0 = self.system.init_state  # Initial State in x form

        if time_space is not None:
            self.system.time = time_space

        state_hat = np.asarray(odeint(
            func=self.state_dot_hat,
            y0=x0,
            t=self.system.time,
            args=(L_gain, k_gain, control_constant,),
            printmessg=verbose
        ))  # Solve Differential Equation

        self.system.states = np.asarray([
            state - (self.system.R.T @ self.system.zeta)
            for state in state_hat
        ]).T
        self.system.d_states = np.asarray([
            self.state_dot_hat(x, 0, L_gain, k_gain, control_constant)
            for x in state_hat
        ]).T

        self.system.controller = - k_gain @ self.system.N @ state_hat.T

    def plot_states(self):
        self.system.plot_states()

    def plot_controller(self):
        self.system.plot_controller()

    def plot_output(self):
        self.system.output()
        self.system.plot_output()
