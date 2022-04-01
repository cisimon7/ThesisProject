import numpy as np
import plotly.graph_objects as go
from control import lqr
from scipy.integrate import odeint
from plotly.subplots import make_subplots
from typing import Optional, List, Tuple, Dict, Any
from LinearStateSpaceModel import LinearStateSpaceModel
from OrthogonalDecomposition import subspaces_from_svd, matrix_rank


class LinearConstraintStateSpaceModel(LinearStateSpaceModel):
    def __init__(self, A: np.ndarray, B: Optional[np.ndarray] = None, C: Optional[np.ndarray] = None,
                 D: Optional[np.ndarray] = None, G: Optional[np.ndarray] = None, F: Optional[np.ndarray] = None,
                 init_state: Optional[np.ndarray] = None):
        """
        Initializes the system state space model and transforming System equation
        taking into account the constraint matrix
        """

        # TODO(Check for controllability of given matrix A and B)

        super().__init__(A, B, C, D, init_state)

        assert (G is not None), "Constraint Matrix not specified. Consider using LinearStateSpaceModel"
        (k, l) = G.shape
        assert (l == self.state_size), "Constraint Matrix cannot be multiplied by state vector"
        self.G = G

        if F is None:
            print("Constraint Jacobian matrix not specified, setting to zero ...")
            self.F = np.zeros_like(A)
        else:
            self.F = F

        (k, l) = self.F.shape
        assert (k == self.state_size), "Constraint Jacobian Matrix cannot be added to state vector change"

        self.reaction_force_size = l

        # Transforming System equation into constraint subspace
        # TODO(Confirm if pseudo-inverse can be used in cases where G is rectangular)
        T = np.eye(self.state_size) - self.F @ (np.linalg.pinv(self.G @ self.F)) @ self.G
        self.A = T @ self.A
        self.B = T @ self.B

        (m, n) = A.shape
        self.state_size = n

        self.rank_G = matrix_rank(self.G)  # TODO(Rank can be retrieved from svd on line 51)
        assert (self.rank_G < self.state_size), \
            f"Invalid Null space size of Constraint matrix: {self.state_size - self.rank_G}"

        row_G, col_G, left_null_G, null_G = subspaces_from_svd(self.G)  # cross-check this function more
        self.N = null_G
        self.R = row_G

        self.zeta = self.R @ init_state  # TODO(To be implemented later)
        self.init_z_state = None

        self.gain = None  # gain for z state
        self.zeta_gain = np.linalg.pinv(self.N @ self.B) @ self.N @ self.A @ self.R.T  # gain for zeta

    def z_dot_gain(self, state: np.ndarray, time: float, gain: np.ndarray) -> np.ndarray:
        """Returns a vector of the state derivative vector at a given state and controller gain"""

        (z, zeta) = (state, self.R @ self.init_state)
        # (z, zeta) = self.N @ state, self.R @ state
        (A, B, N, R) = (self.A, self.B, self.N, self.R)

        self.__assert_state_size(z)
        self.__assert_zeta_size(zeta)
        self.__assert_gain_size(gain)

        # Line 13 from main paper
        _gain_zeta = - np.linalg.pinv(N @ B) @ N @ A @ R.T @ zeta

        # Substituting line 14 into line 9 from the main paper
        result = (N @ (A @ N.T - B @ gain) @ z) + (N @ B @ _gain_zeta) + (N @ A @ R.T @ zeta)

        return result

    def gain_lqr(self, A=None, B=None, Q=None, R=None, set_gain=True):
        N = self.N

        _Q = np.eye(self.state_size - self.rank_G) if (Q is None) else Q
        _R = np.eye(self.control_size) if (R is None) else R

        _A = (N @ self.A @ N.T) if (A is None) else A
        _B = (N @ self.B) if (B is None) else B

        gain = super().gain_lqr(_A, _B, _Q, _R)
        if set_gain:
            self.gain = gain

        return gain

    def ode_gain_solve(self, params: Dict[str, Any] = dict(gain=None), init_state=None,
                       time_space: np.ndarray = np.linspace(0, 10, int(2E3)), verbose=False):

        self.time = time_space

        _init_state = self.init_state if (init_state is None) else init_state
        self.init_z_state = self.N @ _init_state
        self.zeta = self.R @ _init_state

        gain = params['gain']

        # uses identity matrix for Q and R to get an initial lqr gain if gain is not specified
        _gain = self.gain_lqr() if (gain is None) else gain
        self.gain = _gain

        result = odeint(self.z_dot_gain, self.init_z_state, self.time, args=(_gain,), printmessg=verbose)

        z_states = np.asarray(result).transpose()
        d_z_states = np.asarray(
            [self.z_dot_gain(state, time=t, gain=_gain) for (t, state) in zip(self.time, result)]
        ).transpose()  # TODO(Check for better way to finding this from odeint above)

        cons_zeta = self.R.T @ (self.zeta - self.zeta)

        # Calibrating to remove constant zeta gain
        self.states = self.N.T @ z_states + np.asarray([cons_zeta for _ in range(z_states.shape[1])]).T
        self.d_states = self.N.T @ d_z_states

        self.controller = - _gain @ self.N @ self.states
        self.output()

        return self.states, self.d_states

    def plot_states(self, titles=("x states", "x_dot states", "G @ x_dot")):

        assert (self.states is not None), "Run experiment before plotting"
        assert (self.d_states is not None), "Run experiment before plotting"

        fig = make_subplots(rows=1, cols=3, subplot_titles=titles)

        for values in self.states:
            fig.add_trace(
                go.Scatter(x=self.time, y=values, mode='lines'),
                row=1, col=1
            )

        for values in self.d_states:
            fig.add_trace(
                go.Scatter(x=self.time, y=values, mode='lines'),
                row=1, col=2
            )

        for values in (self.G @ self.d_states):
            fig.add_trace(
                go.Scatter(x=self.time, y=values, mode='lines'),
                row=1, col=3
            )

        fig.update_layout(showlegend=False, height=800).show()

    def __assert_control_size(self, control: np.ndarray):
        k = control.shape[0]
        assert (k == self.control_size), "Transformed Control Vector shape error"

    def __assert_gain_size(self, gain: np.ndarray):
        (k, l) = gain.shape
        assert (k == self.control_size), "Transformed Control Vector shape error"
        assert (l == self.state_size - self.rank_G), "Transformed Control Vector shape error"

    def __assert_state_size(self, state: np.ndarray):
        k = state.shape[0]
        assert (k == self.state_size - self.rank_G), "Transformed State Vector shape error"

    def __assert_zeta_size(self, state: np.ndarray):
        k = state.shape[0]
        assert (k == self.rank_G), "Transformed State Vector shape error"
