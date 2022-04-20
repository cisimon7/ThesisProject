import numpy as np
import plotly.graph_objects as go
from typing import Callable, List, Tuple, Dict


class OneDimensionalKalmanFilter:
    def __init__(self, system_dynamic_model_func, variance_extrapolation_func, init_state, init_variance):
        # Initialization step
        self.K_gain = []  # Holds current kalman filter gain
        self.prev_estimate = init_state
        self.prev_variance = [init_variance]
        self.measurement_variance = [init_variance]
        self.sys_model = system_dynamic_model_func
        self.var_extrapolate = variance_extrapolation_func
        self.n = init_state.flatten().shape[0]

    def predict(self):
        # predict next state using state space model
        next_state_pred = self.sys_model(self.prev_estimate)

        # predict state estimate uncertainty
        var_extrap = self.var_extrapolate(self.prev_variance[-1])

        return next_state_pred, var_extrap

    def update(self, measurement, measurement_uncertainty, state_prediction, variance_extrapolation):
        # update Kalman gain
        k_gain = variance_extrapolation / (variance_extrapolation + measurement_uncertainty)
        self.K_gain.append(k_gain)

        # update state estimate
        self.prev_estimate = ((np.eye(self.n) - k_gain) @ state_prediction) + (k_gain @ measurement)

        # update state estimate uncertainty
        variance = (np.eye(self.n) - k_gain) @ self.prev_variance[-1]
        self.prev_variance.append(variance)

    def next_state_estimate(self, measurement: np.ndarray, measurement_uncertainty: np.ndarray):
        self.measurement_variance.append(measurement_uncertainty)

        # loop to run steps while new measurement exist
        (state_pred, var_extrap) = self.predict()
        self.update(measurement, measurement_uncertainty, state_pred, var_extrap)

        return self.prev_estimate

    def plot_kalman_gain(self):
        gains = np.asarray(self.K_gain).flatten()
        count = np.asarray([i for i in np.arange(start=0, stop=len(self.K_gain), step=1)])
        go.Figure(
            data=[go.Scatter(x=count, y=gains, name="True Value")],
            layout=go.Layout(title="Kalman Gain", yaxis=dict(title="Gain"), xaxis=dict(title="Measurement number"))
        ).show()

    def plot_uncertainties(self):
        measurement_uncertainty = np.sqrt(np.asarray(self.prev_variance[1:]).flatten())
        estimate_uncertainty = np.sqrt(np.asarray(self.measurement_variance[1:]).flatten())
        count = np.asarray([i for i in np.arange(start=0, stop=len(self.K_gain), step=1)])
        go.Figure(
            data=[
                go.Scatter(x=count, y=measurement_uncertainty, name="Measurement Uncertainty"),
                go.Scatter(x=count, y=estimate_uncertainty, name="Estimate Uncertainty")
            ],
            layout=go.Layout(title="Uncertainties",
                             yaxis=dict(title="uncertainty"),
                             xaxis=dict(title="Measurement number"))
        ).show()
