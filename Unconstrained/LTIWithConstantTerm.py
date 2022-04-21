import numpy as np
from control import lqr
from scipy.integrate import odeint
import plotly.graph_objects as go


class LTIWithConstantTerm:
    def __init__(self, A: np.ndarray, B: np.ndarray, c: np.ndarray):
        """
        x_dot = Ax + Bu + C
        """
        (self.A, self.B, self.c) = A, B, c
        self.state_size = A.shape[0]
        self.control_size = B.shape[1]

        self.states = None
        self.time_space = None
        self.controller = None

    def dynamics(self, x, t, K_x, K_o):
        A, B, c = self.A, self.B, self.c

        control = -(K_x @ x) - K_o
        x_dot = (A @ x) + (B @ control) + c

        return x_dot

    def lqr_gains(self, Q_xx: np.ndarray, R_uu: np.ndarray):
        """
        L = x.T Q_xx x  +  u.T R_uu u
        J = x.T S_xx x + s_x.T x
        """
        A, B, c = self.A, self.B, self.c
        iR = np.linalg.pinv(R_uu)

        _, S_xx, _ = lqr(A, B, Q_xx, R_uu)
        s_x = - 2 * np.linalg.pinv(A.T - S_xx @ B @ iR @ B.T) @ S_xx @ c

        K_x = iR @ B.T @ S_xx
        K_o = 0.5 * iR @ B.T @ s_x

        return K_x, K_o

    def ode_solve(self, x_init=None, Q_xx: np.ndarray = None, R_uu: np.ndarray = None,
                  time_space=np.linspace(0, 10, int(2E3))):
        self.time_space = time_space
        A, B, c = self.A, self.B, self.c

        Q_xx = np.eye(self.state_size) if Q_xx is None else Q_xx
        R_uu = np.eye(self.control_size) if R_uu is None else R_uu
        x_init = np.round(np.random.randint(1, 10) * np.random.rand(self.state_size),
                              4) if x_init is None else x_init

        K_x, K_o = self.lqr_gains(Q_xx, R_uu)

        x = np.asarray(odeint(
            self.dynamics,
            x_init,
            time_space,
            args=(K_x, K_o)
        ))

        calibration = np.linalg.pinv(A - B @ K_x) @ (B @ K_o - c)
        x = np.asarray([x_ - calibration for x_ in x])

        self.states = x.T
        self.controller = np.asarray([-(K_x @ x_) - K_o for x_ in x]).T

    def plot_states(self, title="State Plot"):
        time = np.linspace(0, self.time_space[-1], 100)
        go.Figure(
            data=[go.Scatter(x=self.time_space, y=state, name=f'output state [{i}]') for (i, state) in
                  enumerate(self.states)] + [
                     go.Scatter(x=time, y=np.zeros_like(time), mode='markers', name='zero')],
            layout=go.Layout(showlegend=True, title=dict(text=title, x=0.5), legend=dict(orientation='h'),
                             xaxis=dict(title='time'), yaxis=dict(title='states'))
        ).show()

    def plot_controller(self, title="Control Plot"):
        go.Figure(
            data=[go.Scatter(x=self.time_space, y=output, name=f'control [{i}]') for (i, output) in
                  enumerate(self.controller)],
            layout=go.Layout(showlegend=True, title=dict(text=title, x=0.5), legend=dict(orientation='h'),
                             xaxis=dict(title='time'), yaxis=dict(title='controller'))
        ).show()
