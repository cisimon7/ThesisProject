import numpy as np
import plotly.graph_objects as go


class ControlSystem:

    def __init__(self, A: np.ndarray, B: np.ndarray, C: np.ndarray = None, D: np.ndarray = None):
        (m, n) = A.shape
        assert (m == n), "System Matrix A not a square matrix"
        self.A = A
        self.state_size = n

        (k, l) = B.shape
        assert (k == n), "System Matrix B cannot be added to system change"
        self.B = B
        self.control_size = l

        if C is not None:
            (k, l) = C.shape
            assert (l == n), "System Matrix C cannot be multiplied by state"
            self.output_size = k

        if D is not None:
            (k, l) = D.shape
            assert (k == self.output_size), ""
            assert (l == self.control_size), ""

        self.C = C
        self.D = D

    def ode_solve(self):
        pass

    def gain_lqr(self):
        pass

    def gain_pole_placement(self):
        pass

    def plot_state(self):
        pass

    def plot_dstate(self):
        pass

    def plot_controller(self):
        pass

    def plot_output(self):
        pass

    def plot_all_states(self):
        pass
