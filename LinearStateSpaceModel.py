import numpy as np
from control import lqr
import plotly.graph_objects as go
from scipy.integrate import odeint
from typing import Dict, Any, Optional
from plotly.subplots import make_subplots


class LinearStateSpaceModel:

    def __init__(self, A: np.ndarray, B: Optional[np.ndarray] = None, C: Optional[np.ndarray] = None,
                 D: Optional[np.ndarray] = None, init_state: Optional[np.ndarray] = None):
        """Initializes the control system state space model and checks all sizes match"""
        (m, n) = A.shape
        assert (m == n), "System Matrix A not a square matrix"
        self.A = A
        self.state_size = n

        self.B = np.zeros_like(self.A) if (B is None) else B
        (k, l) = self.B.shape
        assert (k == n), "System Matrix B cannot be added to system change"
        self.control_size = l

        C = np.eye(self.state_size) if C is None else C

        (k, l) = C.shape
        assert (l == self.state_size), "System Matrix C cannot be multiplied by state vector"
        self.output_size = k
        self.C = C

        D = np.zeros((self.output_size, self.control_size)) if D is None else D
        (k, l) = D.shape
        assert (k == self.output_size), "System Matrix D cannot be added to output change"
        assert (l == self.control_size), "System Matrix D cannot be multiplied by control vector"
        self.D = D

        # Default simulation time
        self.time: np.ndarray = np.linspace(0, 10, int(20))

        # Initial state of the system
        self.init_state: Optional[np.ndarray] = np.zeros(self.state_size) if (init_state is None) else init_state

        self.states: Optional[np.ndarray] = None  # Holds values of state after simulation with odeint
        self.d_states: Optional[np.ndarray] = None  # Holds values of state derivative after simulation with odeint

        self.controller = None
        self.output_states = None

    def xdot(self, state: np.ndarray, time: float, control: np.ndarray) -> np.ndarray:
        """Returns a vector of the state derivative vector at a given state and control input"""

        self.__assert_state_size(state)
        self.__assert_control_size(control)
        return (self.A @ state) + (self.B @ control)

    def x_dot_gain(self, state: np.ndarray, time: float, gain: np.ndarray = None) -> np.ndarray:
        """Returns a vector of the state derivative vector at a given state and controller gain"""

        # uses identity matrix Q and R to get an initial lqr gain if gain is not specified
        _gain = self.gain_lqr() if (gain is None) else gain

        self.__assert_state_size(state)
        self.__assert_control_size(_gain)

        return (self.A - self.B @ _gain) @ state

    def output(self) -> np.ndarray:
        assert (self.controller is not None), "Controller not defined yet"

        result = self.C @ self.states + self.D @ self.controller
        self.output_states = result
        return result

    def gain_lqr(self, A=None, B=None, Q=None, R=None) -> np.ndarray:
        """Returns optimal controller gain given the state and input weight matrix"""

        _Q = np.eye(self.state_size) if (Q is None) else Q  # Initialized to identity matrix to give equal weights
        _R = np.eye(self.control_size) if (R is None) else R  # Initialized to identity matrix to give equal weights

        _A = self.A if (A is None) else A
        _B = self.B if (B is None) else B

        assert (_Q.shape[0] == _Q.shape[1] == _A.shape[0]), "State Weight cannot be multiplied with state vector"
        assert (_R.shape[0] == _R.shape[1] == _B.shape[1]), "Input Weight cannot be multiplied with state vector"

        gain, _, _ = lqr(_A, _B, _Q, _R)  # TODO(remove dependency on LQR using u = - R.inv @ B.T @ S)

        return gain

    def gain_pole_placement(self, pole_locations: np.ndarray):
        pass

    def ode_gain_solve(self, params: Dict[str, Any] = dict(gain=None), init_state=None,
                       time_space: np.ndarray = np.linspace(0, 10, int(2E3)), verbose=False):

        self.time = time_space
        _init_state = self.init_state if (init_state is None) else init_state

        gain = params['gain']
        result = odeint(self.x_dot_gain, _init_state, self.time, args=(gain,), printmessg=verbose)

        self.states = np.asarray(result).transpose()
        self.d_states = np.asarray(
            [self.x_dot_gain(state, time=t, gain=gain) for (t, state) in zip(self.time, result)]
        ).transpose()

        # u = -K x
        self.controller = - gain @ self.states
        self.output()

        return result

    def plot_states(self, titles=("x states", "x_dot states")):

        assert (self.states is not None), "Run experiment before plotting"
        assert (self.d_states is not None), "Run experiment before plotting"

        fig = make_subplots(rows=1, cols=2, subplot_titles=titles)

        for (i, values) in enumerate(self.states):
            fig.add_trace(
                go.Scatter(x=self.time, y=values, mode='lines'),
                row=1, col=1
            )

        for (i, values) in enumerate(self.d_states):
            fig.add_trace(
                go.Scatter(x=self.time, y=values, mode='lines'),
                row=1, col=2
            )

        fig.update_layout(showlegend=False, height=800).show()

    def plot_controller(self, title="Controller Plot"):
        go.Figure(
            data=[go.Scatter(x=self.time, y=cont, name=f'control input [{i}]') for (i, cont) in
                  enumerate(self.controller)],
            layout=go.Layout(showlegend=True, title=dict(text=title, x=0.5), legend=dict(orientation='h'),
                             xaxis=dict(title='time'), yaxis=dict(title='controller size'))
        ).show()

    def plot_output(self, title="Output Plot"):
        go.Figure(
            data=[go.Scatter(x=self.time, y=output, name=f'output state [{i}]') for (i, output) in
                  enumerate(self.output_states)],
            layout=go.Layout(showlegend=True, title=dict(text=title, x=0.5), legend=dict(orientation='h'),
                             xaxis=dict(title='time'), yaxis=dict(title='output states'))
        ).show()

    def __assert_control_size(self, control: np.ndarray):
        k = control.shape[0]
        assert (k == self.control_size), "Control Vector shape error"

    def __assert_gain_size(self, gain: np.ndarray):
        (k, l) = gain.shape
        assert (k == self.control_size), "Gain Vector shape error"
        assert (l == self.state_size), "Gain Vector shape error"

    def __assert_state_size(self, state: np.ndarray):
        k = state.shape[0]
        assert (k == self.state_size), "State Vector shape error"
