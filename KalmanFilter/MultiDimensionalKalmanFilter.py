import numpy as np
import plotly.graph_objects as go
from typing import Any, List, Tuple, Dict, Callable


class MultiDimensionalKalmanFilter:
    def __init__(self, init_state: np.ndarray, init_estimate_uncertainty: np.ndarray,
                 dynamic_model: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
                 covariance_model: Callable[[np.ndarray], np.ndarray],
                 observation_matrix: np.ndarray):
        self.dynamic_model = dynamic_model
        self.covariance_model = covariance_model
        self.observation_matrix = observation_matrix

        self.prev_state = init_state
        self.prev_estimate_cov = init_estimate_uncertainty

        self.kalman_gain = None
        self.n = len(init_state)

        self.control_input = None

    def predict(self):
        # State Extrapolation Equation
        pred_state = self.dynamic_model(self.prev_state, self.control_input, 0)

        # Covariance Extrapolation Equation
        pred_estimate_cov = self.covariance_model(self.prev_estimate_cov)

        return pred_state, pred_estimate_cov

    def next_state_estimate(self, measurement: np.ndarray, measurement_variance: np.ndarray):
        x, P = self.predict()
        H, R = self.observation_matrix, measurement_variance

        # UPDATE EQUATIONS

        # Compute new Kalman gain
        K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
        self.kalman_gain = K

        # Update State Estimate with Measurement
        self.prev_state = x + (K @ (measurement - (H @ x)))

        # Update Estimate uncertainty
        self.prev_estimate_cov = (np.eye(self.n) - K @ H) @ P @ (np.eye(self.n) - K @ H).T + (K @ R @ K.T)

        return self.prev_state
