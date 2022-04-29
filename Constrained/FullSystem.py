import numpy as np
import plotly.graph_objects as go
from control import lqr
from scipy.integrate import odeint
from plotly.subplots import make_subplots
from typing import Optional
from Unconstrained.LinearStateSpaceModel import LinearStateSpaceModel
from OrthogonalDecomposition import subspaces_from_svd, matrix_rank


class FullSystem(LinearStateSpaceModel):
    def __init__(self, A: np.ndarray, B: Optional[np.ndarray] = None, C: Optional[np.ndarray] = None,
                 D: Optional[np.ndarray] = None, G: Optional[np.ndarray] = None, F: Optional[np.ndarray] = None,
                 init_state: Optional[np.ndarray] = None):
        """
        x_dot = Ax + Bu + Fλ
        G x_dot = 0
        y = C @ x

        to be transformed into the form:
        z_dot = (N.T Ac N z) + (N.T B u) + (N.T Ac R ζ)
        x = Nz + Rζ
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

        T = np.eye(self.state_size) - self.F @ (np.linalg.pinv(self.G @ self.F)) @ self.G
        self.A = T @ self.A  # Ac in paper, Equation 3
        self.B = T @ self.B  # Bc in paper, Equation 4

        self.state_size = A.shape[0]
        self.control_size = B.shape[1]

        self.rank_G = matrix_rank(self.G)  # TODO(Rank can be retrieved from svd on line 49)
        assert (self.rank_G < self.state_size), \
            f"Invalid Null space size of G Constraint matrix: {self.state_size - self.rank_G}"

        self.R, _, _, self.N = subspaces_from_svd(self.G)

        # Denotation to avoid repetition
        self.A_nn = self.N @ A @ self.N.T
        self.A_nr = self.N @ A @ self.R.T
        self.B_n = self.N @ B

        self.init_z_state = None
        self.zeta = 1 * self.R @ init_state  # constant

    def dynamics(self, state: np.ndarray, time: float, K_z: np.ndarray, k_0: np.ndarray) -> np.ndarray:
        """Returns a vector of the state derivative vector at a given state and controller gain"""

        (z, zeta) = (state, self.zeta)
        (A_nn, A_nr, B_n) = (self.A_nn, self.A_nr, self.B_n)

        self.__assert_state_size(z)
        self.__assert_zeta_size(zeta)
        self.__assert_gain_size(K_z)

        z_dot = A_nn @ z + (B_n @ ((-K_z @ z) - k_0)) + (A_nr @ zeta)
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

        R = 0.00001 * R

        iR = np.linalg.pinv(R)

        _, S_nn, _ = lqr(A_nn, B_n, Q, R)
        phi = - np.linalg.pinv(A_nn.T - S_nn @ B_n @ iR @ B_n.T) @ S_nn @ A_nr @ zeta

        k_z = iR @ B_n.T @ S_nn
        k_0 = iR @ B_n.T @ phi

        return k_z, k_0

    def ode_solve(self, k_z=None, k_0=None, init_state=None, time_space: np.ndarray = np.linspace(0, 10, int(2E3)),
                  verbose=False):

        self.time = time_space

        _init_state = self.init_state if (init_state is None) else init_state
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

        z_states = result.T
        d_z_states = np.asarray([
            self.dynamics(state, time=t, K_z=k_z, k_0=k_0)
            for (t, state) in zip(self.time, result)
        ]).transpose()

        # cons_zeta = self.R.T @ self.zeta * 0  # TODO(Adding zeta makes state not tend to zero)
        # self.states = self.N.T @ z_states + np.asarray([cons_zeta for _ in range(z_states.shape[1])]).T

        self.states = z_states
        self.d_states = self.N.T @ d_z_states

        # self.controller = np.asarray([- (k_z @ z) + k_0 for z in z_states.T]).T
        # self.output()

        return self.states, self.d_states

    def pplot_states(self):
        go.Figure(
            data=[
                go.Scatter(x=self.time, y=values, mode='lines', name=f"state-{i}")
                for (i, values) in enumerate(self.states)
            ],
            layout=go.Layout(title=dict(text="z states", x=0.5), xaxis=dict(title='time'),
                             yaxis=dict(title='states'))
        ).show()

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
