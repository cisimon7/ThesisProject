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
        self.system = system
        (z, zeta) = (system.init_state, system.zeta)
        (N, R) = (system.N, system.R)
        (A, B) = (system.A, system.B)
        (C, D) = (system.C, system.D)
        (r, l) = (system.rank_G, system.state_size - system.rank_G)

        # (l, r)
        self.init_state = np.block([[system.init_state, system.zeta]])

        self.A = np.block([
            [N @ A @ N.T, N @ A @ R],
            [np.zeros((1, l)), np.zeros((1, r))]
        ])

        self.B = np.block([
            [N @ B],
            [np.zeros((1, r))]
        ])

        self.C = C @ np.block([[N, R]])

        self.D = system.D

    def z_dot_hat_gain(self, estimator_state: EstimatorState, time: float, gain: np.ndarray) -> EstimatorState:
        """Returns an estimate of the state vector derivative vector at a given state and controller gain"""

        (state_hat, output) = estimator_state
        (A, B, C, D) = (self.A, self.B, self.C, self.D)
        state_estimate = A @ state_hat + B + gain @ (output - C @ state_hat)

        # In reality, this is gotten freely
        y = self.system.states[np.where(self.system.time == time)]  # + noise
        return EstimatorState(state=state_estimate, output=y)

    def estimator(self, params: ZDotHatParams = ZDotHatParams(gain=None), verbose=False):
        _gain = lqr(
            self.A,
            self.C,
            np.eye(self.A.shape[0]),
            np.eye(self.C.shape[1])
        ) if params.gain is None else params.gain

        self.system.ode_gain_solve()

        state_hat = odeint(
            func=self.z_dot_hat_gain,
            y0=EstimatorState(state=self.init_state, output=np.zeros(self.system.output_size)),
            t=self.system.time,
            args=(_gain,),
            printmessg=verbose
        )

        go.Figure(
            data=[go.Scatter(x=self.system.time, y=state) for state in state_hat]
        ).show()

