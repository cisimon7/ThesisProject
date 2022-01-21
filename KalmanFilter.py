import numpy as np
from typing import Callable, List, Tuple, Dict


class KalmanFilter:
    def __init__(self, system_dynamic_model_func, variance_extrapolation_func, init_state, init_variance):
        # Initialization step
        self.K_gain = None  # Holds current kalman filter gain
        self.prev_estimate = init_state
        self.prev_variance = init_variance
        self.sys_model = system_dynamic_model_func
        self.var_extrapolate = variance_extrapolation_func
        self.n = init_state.flatten().shape[0]

    def predict(self):
        # predict next state using state space model
        next_state_pred = self.sys_model(self.prev_estimate)

        # predict state estimate uncertainty
        var_extrap = self.var_extrapolate(self.prev_variance)

        return next_state_pred, var_extrap

    def update(self, measurement, measurement_uncertainty, state_prediction, variance_extrapolation):
        # update Kalman gain
        self.K_gain = variance_extrapolation / (variance_extrapolation + measurement_uncertainty)

        # update state estimate
        self.prev_estimate = ((np.eye(self.n) - self.K_gain) @ state_prediction) + (self.K_gain @ measurement)

        # update state estimate uncertainty
        self.prev_variance = (np.eye(self.n) - self.K_gain) @ self.prev_variance

    def next_state_estimate(self, measurement: np.ndarray, measurement_uncertainty: np.ndarray):
        # loop to run steps while new measurement exist
        (state_pred, var_extrap) = self.predict()
        self.update(measurement, measurement_uncertainty, state_pred, var_extrap)

        return self.prev_estimate, self.prev_variance
