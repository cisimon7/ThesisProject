from dataclasses import dataclass
from typing import Optional

import numpy as np
from control import lqe
from scipy.integrate import odeint

from LinearConstraintStateSpaceModel import LinearConstraintStateSpaceModel


@dataclass
class ZDotHatParams:
    """Class for storing parameters for the function z_dot_hat_gain to be passed to odeint"""
    gain: Optional[np.ndarray]


class LTIConstraintStateEstimator:

    def __init__(self, system: LinearConstraintStateSpaceModel):
        self.system = system
        (N, R) = (system.N, system.R)
        (A, B) = (system.A, system.B)
        (C, D) = (system.C, system.D)
        (self.r, self.l) = (system.rank_G, system.state_size - system.rank_G)

        (r, l) = self.r, self.l

        self.A = np.block([
            [N @ A @ N.T, N @ A @ R.T],
            [np.zeros((r, l)), np.zeros((r, r))]
        ])

        self.B = np.block([
            [N @ B],
            [np.zeros((r, system.control_size))]
        ])

        self.C = C @ np.block([[N.T, R.T]])

        self.D = np.block([
            [system.D]
        ])

        # self.system.A = self.A
        # self.system.B = self.B
        # self.system.C = self.C
        # self.system.D = self.D

        self.zeta = R @ self.system.init_state

    def state_dot_hat(self, x_state: np.ndarray, time, L_gain: np.ndarray, K_gain: np.ndarray) -> np.ndarray:
        """Returns an estimate of the state vector derivative vector at a given state and controller gain"""

        (A, B, C, D) = (self.A, self.B, self.C, self.D)
        (N, R) = (self.system.N, self.system.R)

        z_hat = N @ x_state  # z state
        zeta_hat = R @ x_state  # zeta state
        state_hat = np.r_[z_hat, zeta_hat]

        # To be received from measurement, contains errors
        y_actual = (C @ state_hat)

        sigma = 1
        mu = 1
        noise = sigma * np.random.randn(y_actual.shape[0]) + mu

        y_actual = y_actual + 0 * noise  # Adding noise to simulate sensor noise

        U_z = - K_gain @ z_hat
        U_zeta = - self.system.zeta_gain @ self.system.zeta

        es_B = np.block([[B, L_gain]])
        es_input = np.vstack((
            (U_z + U_zeta).reshape(-1, 1), y_actual.reshape(-1, 1)
        ))

        d_state_estimate = ((A - L_gain @ C) @ state_hat) + (es_B @ es_input).ravel()

        d_z_ = d_state_estimate[:self.l]
        d_zeta_ = d_state_estimate[self.l:]

        d_x = N.T @ d_z_

        return d_x

    def estimate(self, params: ZDotHatParams = ZDotHatParams(gain=None), time_space=None, verbose=False):
        L_gain, _, _ = lqe(
            self.A,
            np.eye(self.A.shape[0]),
            self.C,
            np.eye(self.A.shape[0]),
            np.eye(self.C.shape[1])
        ) if params.gain is None else params.gain  # Estimator Gain

        k_gain = self.system.gain_lqr()  # Controller Gain
        x0 = self.system.init_state  # Initial State in x forms

        if time_space is not None:
            self.system.time = time_space

        state_hat = np.asarray(odeint(
            func=self.state_dot_hat,
            y0=x0,
            t=self.system.time,
            args=(L_gain, k_gain,),
            printmessg=verbose
        ))  # Solve Differential Equation

        self.system.states = np.asarray([state - (self.system.R.T @ self.system.zeta) for state in state_hat]).T
        self.system.d_states = np.asarray([self.state_dot_hat(x, 0, L_gain, k_gain) for x in state_hat]).T

        self.system.controller = - k_gain @ self.system.N @ state_hat.T

    def plot_states(self):
        self.system.plot_states()

    def plot_controller(self):
        self.system.plot_controller()

    def plot_output(self):
        self.system.output()
        self.system.plot_output()
