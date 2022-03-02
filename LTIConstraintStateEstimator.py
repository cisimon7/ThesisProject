import numpy as np
from control import lqr, lqe
from dataclasses import dataclass
from scipy.integrate import odeint
from typing import List, Tuple, Dict, Any, Optional
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from LinearConstraintStateSpaceModel import LinearConstraintStateSpaceModel


@dataclass
class EstimatorState:
    """Class for keeping track of estimator input states"""
    state: np.ndarray  # represents the current state estimates
    output: np.ndarray  # represents the current state output


@dataclass
class ZDotHatParams:
    """Class for storing parameters for the function z_dot_hat_gain to be passed to odeint"""
    gain: Optional[np.ndarray]


class LTIConstraintStateEstimator:

    def __init__(self, system: LinearConstraintStateSpaceModel):
        system.ode_gain_solve()
        self.system = system
        (N, R) = (system.N, system.R)
        (A, B) = (system.A, system.B)
        (C, D) = (system.C, system.D)
        (r, l) = (system.rank_G, system.state_size - system.rank_G)

        self.r = r
        self.l = l

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
            [system.D],
            [np.zeros((r, system.control_size))]
        ])

        self.output = np.array([])
        self.t_prev = -self.system.time[0]

    def state_dot_hat(self, init_state: np.ndarray, time, L_gain: np.ndarray, K_gain: np.ndarray) -> np.ndarray:
        """Returns an estimate of the state vector derivative vector at a given state and controller gain"""

        (A, B, C, D) = (self.A, self.B, self.C, self.D)
        (Ac, Bc) = (self.system.A, self.system.B)
        (N, R) = (self.system.N, self.system.R)

        z_hat = N @ init_state  # z state
        zeta_hat = R @ init_state  # zeta state
        state_hat = np.r_[z_hat, zeta_hat]

        # To be received from measurement, contains errors
        y_actual = (C @ init_state) # TODO(Should be state_hat instead of init_state)

        U_z = - K_gain @ N @ state_hat
        U_zeta = - np.linalg.pinv(N@Bc) @ N @ Ac @ R.T @ zeta_hat

        es_B = np.block([[B, L_gain]])
        es_input = np.vstack((
            (U_z + U_zeta).reshape(-1, 1), y_actual.reshape(-1, 1)
        ))

        # print((L_gain@y_actual)-(L_gain @ C @ state_hat))

        state_estimate = ((A - L_gain @ C) @ state_hat) + (es_B @ es_input).ravel()

        z_ = state_estimate[:self.l]
        zeta_ = state_estimate[self.l:]

        x = (N.T @ z_) + (R.T @ zeta_)

        return x

    def estimate(self, params: ZDotHatParams = ZDotHatParams(gain=None), verbose=False):
        L_gain, _, _ = lqe(
            self.A,
            np.eye(self.A.shape[0]),
            self.C,
            np.eye(self.A.shape[0]),
            np.eye(self.C.shape[1])
        ) if params.gain is None else params.gain  # Estimator Gain
        k_gain = self.system.gain_lqr()  # Controller Gain
        x0 = self.system.init_state  # Initial State in x forms

        state_hat = odeint(
            func=self.state_dot_hat,
            y0=x0,
            t=self.system.time,
            args=(L_gain, k_gain,),
            printmessg=verbose
        )  # Solve Differential Equation

        # Plot States
        go.Figure(
            data=[go.Scatter(x=self.system.time, y=state, mode='lines') for state in state_hat.T],
            layout=go.Layout()
        ).show()

        # go.Figure(
        #     data=[go.Scatter(x=self.system.time, y=values, mode='lines') for values in self.system.states],
        #     layout=go.Layout()
        # ).show()

        # go.Figure(
        #     data=[go.Scatter(x=self.system.time, y=output, name=f'output state [{i}]') for (i, output) in
        #           enumerate((self.system.C @ state_hat.T))],
        #     layout=go.Layout(showlegend=True, title="Estimate Output", legend=dict(orientation='h'),
        #                      xaxis=dict(title='time'), yaxis=dict(title='output states'))
        # ).show()

        # self.system.plot_output()
