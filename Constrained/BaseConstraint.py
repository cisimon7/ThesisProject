import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Optional
from Unconstrained.LinearStateSpaceModel import LinearStateSpaceModel
from OrthogonalDecomposition import subspaces_from_svd, matrix_rank


class BaseConstraint(LinearStateSpaceModel):
    def __init__(self, A: np.ndarray, B: Optional[np.ndarray] = None, C: Optional[np.ndarray] = None,
                 D: Optional[np.ndarray] = None, G: Optional[np.ndarray] = None, F: Optional[np.ndarray] = None,
                 g=None, init_state: Optional[np.ndarray] = None, x_desired: Optional[np.ndarray] = None,
                 dx_desired: Optional[np.ndarray] = None):
        """
        x_dot = Ax + Bu + Fλ
        G x_dot = 0
        y = C @ x

        to be transformed into the form:
        z_dot = (N.T Ac N z) + (N.T B u) + (N.T Ac R ζ)
        x = Nz + Rζ
        """
        # Runs the init method of the super class LinearStateSpaceModel
        super().__init__(A, B, C, D, init_state, x_desired, dx_desired)

        self.z_states = None
        self.dz_states = None
        self.init_z_state = None

        assert (G is not None), "Constraint Matrix not specified. Consider using LinearStateSpaceModel"
        (k, l) = G.shape
        assert (l == self.state_size), "Constraint Matrix cannot be multiplied by state vector"
        self.constraint_size = k
        self.G = G

        g = np.zeros(self.state_size).ravel() if g is None else g.ravel()

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
        self.g = T @ g

        self.state_size = A.shape[0]
        self.control_size = B.shape[1]

        self.rank_G = matrix_rank(self.G)  # TODO(Rank can be retrieved from svd on line 52)
        assert (self.rank_G < self.state_size), \
            f"Invalid Null space size of G Constraint matrix: {self.state_size - self.rank_G}"

        self.R, _, _, self.N = subspaces_from_svd(self.G)

        self.zeta = self.R @ self.init_state  # constant

    def dynamics(self, state: np.ndarray, time: float, K_z: np.ndarray, k_0: np.ndarray):
        pass

    # def output(self, states=None) -> np.ndarray:
    #     assert (self.controller is not None), "Controller not defined yet"
    #
    #     states = self.states if (states is None) else states
    #
    #     result = (self.C @ states) + (self.D @ self.controller) /
    #     - (self.C @ self.R.T @ np.asarray([self.zeta for _ in states.T]).T)
    #     self.output_states = result
    #     return result

    def z_output(self, z_states=None) -> np.ndarray:
        assert (self.controller is not None), "Controller not defined yet"

        z_states = self.z_states if (z_states is None) else z_states

        result = (self.C @ self.N.T @ z_states) + (self.D @ self.controller)
        return result

    def plot_states(self, title=(), state_name="z-state", width=None, height=None):
        go.Figure(
            data=[
                     go.Scatter(x=self.time, y=values, mode='lines', name=f"{state_name}[{i}]",
                                marker_color=px.colors.qualitative.Dark24[i])
                     for (i, values) in enumerate(self.z_states)
                 ] + [
                     go.Scatter(x=self.time, y=[values for _ in self.time], line=dict(width=2, dash='5px'),
                                name=f"desired [{i}]", marker_color=px.colors.qualitative.Dark24[i])
                     for (i, values) in enumerate(self.N @ self.x_desired)
                 ],
            layout=go.Layout(showlegend=True, title=dict(text="z states", x=0.5), legend=dict(orientation='h'),
                             xaxis=dict(title='time'), yaxis=dict(title='z - states'), width=width, height=height)
        ).show()

    def plot_x_states(self, title=(), interest=None):
        go.Figure(
            data=[
                     go.Scatter(x=self.time, y=values, mode='lines', name=f"x-state [{i}]",
                                marker_color=px.colors.qualitative.Dark24[i%24])
                     for (i, values) in enumerate(self.states if interest is None else self.states[interest[0]:interest[1]])
                 ] +
                 [
                     go.Scatter(x=self.time, y=[values for _ in self.time], line=dict(width=2, dash='5px'),
                                name=f"desired [{i}]", marker_color=px.colors.qualitative.Dark24[i%24])
                     for (i, values) in enumerate(self.x_desired)
                 ],
            layout=go.Layout(showlegend=False, title=dict(text="x states", x=0.5), legend=dict(orientation='h'),
                             xaxis=dict(title='time'), yaxis=dict(title='x - states'))
        ).show()

    def plot_d_states(self, title=()):
        go.Figure(
            data=[
                     go.Scatter(x=self.time, y=values, mode='lines', name=f"dz-state [{i}]",
                                marker_color=px.colors.qualitative.Dark24[i%24])
                     for (i, values) in enumerate(self.dz_states)
                 ] + [
                     go.Scatter(x=self.time, y=[values for _ in self.time], line=dict(width=2, dash='5px'),
                                name=f"desired [{i}]",
                                marker_color=px.colors.qualitative.Dark24[i%24])
                     for (i, values) in enumerate(self.N @ self.dx_desired)
                 ],
            layout=go.Layout(showlegend=True, title=dict(text="dz states", x=0.5), legend=dict(orientation='h'),
                             xaxis=dict(title='time'), yaxis=dict(title='dz - states'))
        ).show()

    def plot_z_output(self, title=()):
        z_out = self.z_output()
        go.Figure(
            data=[
                     go.Scatter(x=self.time, y=values, mode='lines', name=f"z-state [{i}]",
                                marker_color=px.colors.qualitative.Dark24[i%24])
                     for (i, values) in enumerate(z_out)
                 ] + [
                     go.Scatter(x=self.time, y=values, line=dict(width=2, dash='5px'), name=f"desired [{i}]",
                                marker_color=px.colors.qualitative.Dark24[i%24])
                     for (i, values) in
                     enumerate(self.z_output(np.asarray([self.N @ self.x_desired for _ in self.time]).T))
                 ],
            layout=go.Layout(showlegend=True, title=dict(text="z output states", x=0.5), legend=dict(orientation='h'),
                             xaxis=dict(title='time'), yaxis=dict(title='z - states'))
        ).show()

    def plot_overview(self, titles=("x states", "x_dot states", "G @ x_dot")):

        assert (self.states is not None), "Run experiment before plotting"
        assert (self.d_states is not None), "Run experiment before plotting"

        fig = make_subplots(rows=1, cols=3, subplot_titles=titles)

        for values in self.states:
            fig.add_trace(
                go.Scatter(x=self.time, y=values, mode='lines'),
                row=1, col=1
            )

        for values in self.x_desired:
            fig.add_trace(
                go.Scatter(x=self.time, y=[values for _ in self.time], line=dict(width=2, dash='5px')),
                row=1, col=1
            )

        for values in self.d_states:
            fig.add_trace(
                go.Scatter(x=self.time, y=values, mode='lines'),
                row=1, col=2
            )

        for values in self.dx_desired:
            fig.add_trace(
                go.Scatter(x=self.time, y=[values for _ in self.time], line=dict(width=2, dash='5px')),
                row=1, col=2
            )

        for values in (self.G @ self.d_states):
            fig.add_trace(
                go.Scatter(x=self.time, y=values, mode='lines'),
                row=1, col=3
            )

        fig.update_layout(showlegend=False, height=800).show()

    def assert_control_size(self, control: np.ndarray):
        k = control.shape[0]
        assert (k == self.control_size), "Transformed Control Vector shape error"

    def assert_gain_size(self, gain: np.ndarray):
        (k, l) = gain.shape
        assert (k == self.control_size), "Transformed Control Vector shape error"
        assert (l == self.state_size - self.rank_G), "Transformed Control Vector shape error"

    def assert_state_size(self, state: np.ndarray):
        k = state.shape[0]
        assert (k == self.state_size - self.rank_G), "Transformed State Vector shape error"

    def assert_zeta_size(self, state: np.ndarray):
        k = state.shape[0]
        assert (k == self.rank_G), "Transformed State Vector shape error"
