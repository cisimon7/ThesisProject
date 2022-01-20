import numpy as np
from control import lqr
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
        (z, zeta) = (system.init_state, system.zeta)
        (N, R) = (system.N, system.R)
        (A, B) = (system.A, system.B)
        (C, D) = (system.C, system.D)
        (r, l) = (system.rank_G, system.state_size - system.rank_G)

        # (l, r)
        self.init_state = np.block([[system.init_state, system.zeta]])

        self.A = np.block([
            [N @ A @ N.T, N @ A @ R.T],
            [np.zeros((r, l)), np.zeros((r, r))]
        ])

        self.B = np.block([
            [N @ B],
            [np.zeros((r, system.control_size))]
        ])

        self.C = C @ np.block([[N.T, R.T]])

        self.D = system.D

        self.init_state = np.r_[system.init_z_state, system.zeta]
        self.x_prev = self.system.init_state

        self.output = np.array([])
        self.t_prev = -self.system.time[0]

    def z_dot_hat_gain(self, estimator_input: np.ndarray, time, gain: np.ndarray) -> np.ndarray:
        """Returns an estimate of the state vector derivative vector at a given state and controller gain"""

        (state_hat, y_actual) = (estimator_input[:self.system.state_size], estimator_input[self.system.state_size:])
        (A, B, C, D) = (self.A, self.B, self.C, self.D)

        control_ff = - self.system.gain @ state_hat[:self.system.state_size - self.system.rank_G]

        state_estimate = (A @ state_hat) + (B @ control_ff) + gain @ (y_actual - C @ state_hat)

        x = self.system.state_time_solve(init_state=self.x_prev.ravel(), prev_time=self.t_prev, current_time=time)
        x = x.T[-1]
        self.x_prev = x

        # Won't be derived in real life, it is measured, and it contains errors
        y_actual = (self.system.C @ x) - (self.system.D @ self.system.gain @ x[:self.system.state_size - self.system.rank_G])
        self.output = np.vstack((self.output, y_actual.flatten()))

        result = np.r_[state_estimate, y_actual.ravel()]
        return result

    def estimate(self, params: ZDotHatParams = ZDotHatParams(gain=None), verbose=False):
        _gain, _, _ = lqr(
            self.A,
            self.C,
            np.eye(self.A.shape[0]),
            np.eye(self.C.shape[1])
        ) if params.gain is None else params.gain

        x0 = self.system.init_state
        control_ff = - self.system.gain @ x0[:self.system.state_size - self.system.rank_G]
        y0 = (self.system.C @ x0) + (self.system.D @ control_ff)
        self.output = y0

        state_hat = odeint(
            func=self.z_dot_hat_gain,
            y0=np.r_[x0, y0],
            t=self.system.time,
            args=(_gain,),
            printmessg=verbose
        )

        # go.Figure(
        #     data=[go.Scatter(x=self.system.time, y=state, mode='lines') for state in
        #           state_hat.T[:self.system.state_size]],
        #     layout=go.Layout()
        # ).show()
        #
        # go.Figure(
        #     data=[go.Scatter(x=self.system.time, y=values, mode='lines') for
        #           values in self.system.states],
        #     layout=go.Layout()
        # ).show()

        go.Figure(
            data=[go.Scatter(x=self.system.time, y=output, name=f'output state [{i}]') for (i, output) in
                  enumerate(self.output.T)],
            layout=go.Layout(showlegend=True, title="Estimate Output", legend=dict(orientation='h'),
                             xaxis=dict(title='time'), yaxis=dict(title='output states'))
        ).show()

        self.system.plot_output()
