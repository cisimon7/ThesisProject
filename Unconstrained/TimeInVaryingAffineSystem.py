import numpy as np
from control import lqr
from scipy.integrate import odeint
import plotly.graph_objects as go


class TimeInVaryingAffineSystem:
    def __init__(self, A: np.ndarray, B: np.ndarray, C: np.ndarray):
        """
        x_dot = Ax + Bu + C
        """
        (self.A, self.B, self.C) = A, B, C
        self.state_size = A.shape[0]
        self.control_size = B.shape[1]

        self.states = None
        self.time_space = None
        self.controller = None

    def dynamics(self, x, t, K_x, K_o):
        A, B, C = self.A, self.B, self.C

        control = -(K_x @ x) - K_o
        x_dot = (A @ x) + (B @ control) + C

        return x_dot

    def lqr_gains(self, Q_xx: np.ndarray, q_x: np.ndarray, R_uu: np.ndarray, r_u: np.ndarray):
        """
        L = [x, 1][[Q_xx, q_x],  [[x],   [u, 1][[R_uu, r_u],  [[u],
                   [q_x.T, q_o]]  [1]]          [r_u.T, r_o]]  [1]]

        J = [x, 1][[S_xx, s_x],  [[x],
                   [s_x.T, s_o]]  [1]]
        """
        A, B, C = self.A, self.B, self.C
        iR = np.linalg.pinv(R_uu)

        _, S_xx, _ = lqr(A, B, Q_xx, R_uu)
        s_x = np.linalg.pinv(A.T - S_xx @ B @ iR @ B.T) @ ((S_xx @ B @ iR @ r_u) - q_x - (S_xx @ C))

        K_x = iR @ S_xx @ B
        K_o = iR @ (r_u + (s_x.T @ B))

        return K_x, K_o

    def ode_solve(self, x_init=None, Q_xx: np.ndarray = None, q_x: np.ndarray = None, R_uu: np.ndarray = None,
                  r_u: np.ndarray = None, time_space=np.linspace(0, 10, int(2E3))):
        self.time_space = time_space
        Q_xx = np.eye(self.state_size) if Q_xx is None else Q_xx
        R_uu = np.eye(self.control_size) if R_uu is None else R_uu
        q_x = np.ones(self.state_size).T if q_x is None else q_x
        r_u = np.ones(self.control_size).T if r_u is None else r_u
        x_init = np.round(np.random.randint(1, 10) * np.random.rand(self.state_size), 4) if x_init is None else x_init

        K_x, K_o = self.lqr_gains(Q_xx, q_x, R_uu, r_u)
        K_o = K_o * 0

        x = np.asarray(odeint(
            self.dynamics,
            x_init,
            time_space,
            args=(K_x, K_o)
        ))
        self.states = x.T
        self.controller = np.asarray([-(K_x @ x_) - K_o for x_ in x]).T

    def plot_states(self, title="State Plot"):
        time = np.linspace(0, self.time_space[-1], 100)
        go.Figure(
            data=[go.Scatter(x=self.time_space, y=state, name=f'output state [{i}]') for (i, state) in
                  enumerate(self.states)] + [
                     go.Scatter(x=time, y=np.zeros_like(time), mode='markers')],
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
