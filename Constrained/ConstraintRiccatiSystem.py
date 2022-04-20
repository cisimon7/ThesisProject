import numpy as np
import plotly.graph_objects as go
from control import lqr
from scipy.integrate import odeint
from plotly.subplots import make_subplots
from typing import Optional
from Unconstrained.LinearStateSpaceModel import LinearStateSpaceModel
from OrthogonalDecomposition import subspaces_from_svd, matrix_rank


class ConstraintRiccatiSystem(LinearStateSpaceModel):
    def __init__(self, A: np.ndarray, B: Optional[np.ndarray] = None, C: Optional[np.ndarray] = None,
                 D: Optional[np.ndarray] = None, G: Optional[np.ndarray] = None, F: Optional[np.ndarray] = None,
                 init_state: Optional[np.ndarray] = None):
        """
        Initializes the system state space model and transforming System equation
        taking into account the constraint matrix
        """

        super().__init__(A, B, C, D, init_state)  # Runs the init method of the super class LinearStateSpaceModel

        assert (G is not None), "Constraint Matrix not specified. Consider using LinearStateSpaceModel"
        (k, l) = G.shape
        assert (l == self.state_size), "Constraint Matrix cannot be multiplied by state vector"
        self.constraint_size = k
        self.G = G

        if F is None:
            print("Constraint Jacobian matrix not specified, setting to zero ...")
            self.F = np.zeros_like(A)
        else:
            self.F = F

        (k, l) = self.F.shape
        assert (k == self.state_size), "Constraint Jacobian Matrix cannot be added to state vector change"
        self.reaction_force_size = l

        # TODO(Confirm if pseudo-inverse can be used in cases where G@F is rectangular)
        T = np.eye(self.state_size) - self.F @ (np.linalg.pinv(self.G @ self.F)) @ self.G
        self.A = T @ self.A  # Ac in paper, Equation 3
        self.B = T @ self.B  # Bc in paper, Equation 4

        self.state_size = A.shape[0]
        self.control_size = B.shape[1]

        self.rank_G = matrix_rank(self.G)  # TODO(Rank can be retrieved from svd on line 52)
        assert (self.rank_G < self.state_size), \
            f"Invalid Null space size of G Constraint matrix: {self.state_size - self.rank_G}"

        self.R, _, _, self.N = subspaces_from_svd(self.G)

        self.init_z_state = None
        self.zeta = self.R @ init_state  # constant

    def z_dot_gain(self, state: np.ndarray, time: float, gain: np.ndarray, control_const: np.ndarray) -> np.ndarray:
        """Returns a vector of the state derivative vector at a given state and controller gain"""

        (z, zeta) = (state, self.zeta)
        (A, B, N, R) = (self.A, self.B, self.N, self.R)

        self.__assert_state_size(z)
        self.__assert_zeta_size(zeta)
        self.__assert_gain_size(gain)

        # Substituting line 14 into line 9 from the main paper
        result = (N @ (A @ N.T - B @ gain) @ z) + (N @ B @ control_const) + (N @ A @ R.T @ zeta)

        return result

    def riccati_solve(self, Qz=None, Ru=None):
        (Ac, Bc, zeta) = (self.A, self.B, self.zeta)
        (N, R) = (self.N, self.R)
        (A_nn, A_nr, B_n) = (N @ Ac @ N.T, N @ Ac @ R.T, N @ Bc)

        Qz = np.eye(A_nn.shape[0]) if Qz is None else Qz
        Ru = np.eye(B_n.shape[1]) if Ru is None else Ru

        iR = np.linalg.pinv(Ru)

        _, S, _ = lqr(A_nn, B_n, Qz, Ru)
        s_z = - np.linalg.pinv((-0.5 * S @ B_n @ iR @ B_n.T) + (0.5 * A_nn)) @ (S @ A_nr @ zeta)

        K_z = iR @ B_n.T @ S
        const = - iR @ B_n.T @ s_z  # - np.linalg.pinv(B_n) @ A_nr @ zeta

        return K_z, const, S, s_z

    def ode_gain_solve(self, _gain=None, control_const=None, init_state=None,
                       time_space: np.ndarray = np.linspace(0, 10, int(2E3)), verbose=False):

        self.time = time_space

        _init_state = self.init_state if (init_state is None) else init_state
        self.init_z_state = self.N @ _init_state
        self.zeta = self.R @ _init_state

        if _gain is None:
            _gain, control_const, _, _ = self.riccati_solve()

        result = odeint(self.z_dot_gain, self.init_z_state, self.time, args=(_gain, control_const), printmessg=verbose)

        z_states = np.asarray(result).transpose()
        d_z_states = np.asarray(
            [self.z_dot_gain(state, time=t, gain=_gain, control_const=control_const) for (t, state) in
             zip(self.time, result)]
        ).transpose()

        cons_zeta = self.R.T @ self.zeta * 0  # TODO(Adding zeta makes state not tend to zero)

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
