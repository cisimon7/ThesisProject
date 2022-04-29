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
        zeta_dot = 0
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
        (self.r, self.l) = (self.rank_G, self.state_size - self.rank_G)
        (r, l) = (self.r, self.l)

        # Denotation to avoid repetition
        self.A_nn = self.N @ A @ self.N.T
        self.A_nr = self.N @ A @ self.R.T
        self.B_n = self.N @ B

        # New A matrix
        self.A_z_zeta = np.block([
            [self.A_nn, self.A_nr],
            [np.zeros((r, l)), np.zeros((r, r))]
        ])

        # New B matrix
        self.B_z_zeta = np.block([
            [self.B_n],
            [np.zeros((r, self.control_size))]
        ])

        self.init_z_state = None
        self.zeta = self.R @ init_state

    def dynamics(self, state: np.ndarray, time: float, K_z: np.ndarray, k_0: np.ndarray) -> np.ndarray:
        """
        :param state: a column vector of z and zeta states
        :param time: needed for odeint, not needed for LTI systems
        :param K_z: controller gain
        :param k_0: controller constant gain
        :return:
        """
        A, B = self.A_z_zeta, self.B_z_zeta

        z_dot = A @ state + (B @ ((-K_z @ state) - k_0))
        return z_dot

    def riccati_solve(self, Q=None, R=None):
        A, B = self.A_z_zeta, self.B_z_zeta
        (r, l) = (self.r, self.l)

        # Q = np.block([
        #     [np.eye(l), np.zeros((l, r))],
        #     [np.zeros((r, l)), np.zeros((r, r))]
        # ]) if Q is None else Q
        Q = np.eye(A.shape[0])
        R = np.eye(B.shape[1]) if R is None else R

        R = R

        iR = np.linalg.pinv(R)

        _, S_z_zeta, _ = lqr(np.ones_like(A), np.ones_like(B), Q, R)

        k_z = iR @ B.T @ S_z_zeta
        k_0 = np.zeros((B.shape[1], 1))

        return k_z, k_0

    def ode_solve(self, k_z=None, k_0=None, init_state=None, time_space=np.linspace(0, 10, int(2E3)), verbose=False):
        A, B = self.A_z_zeta, self.B_z_zeta
        R, N = self.R, self.N
        r, l = self.r, self.l

        self.time = time_space

        _init_state = self.init_state if (init_state is None) else init_state
        self.init_z_state = self.N @ _init_state
        self.zeta = self.R @ _init_state

        init = np.r_[self.init_z_state, self.zeta]

        (k_z, k_0) = self.riccati_solve() if (k_z is None or k_0 is None) else (k_z, k_0)

        result = np.asarray(odeint(
            self.dynamics,
            init,
            self.time,
            args=(k_z, k_0),
            printmessg=verbose
        ))

        z_zeta = result
        x = np.block([[N.T, R.T]]) @ z_zeta
        self.states = x

        # self.d_states = self.N.T @ d_z_states

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
