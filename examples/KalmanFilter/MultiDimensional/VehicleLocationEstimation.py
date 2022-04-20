import numpy as np
import plotly.graph_objects as go

from KalmanFilter.MultiDimensionalKalmanFilter import MultiDimensionalKalmanFilter

if __name__ == "__main__":
    """
    VEHICLE LOCATION ESTIMATION
    In this example, we estimate the location of the vehicle in the XY plane.
    The vehicle has an onboard location sensor that reports X and Y coordinates of the system.
    We assume constant acceleration dynamics.
    system state described as: state = [ x, dx, ddx, y, dy, ddy ]
    """

    delta_time = 1  # Constant time interval for taking measurement
    acceleration_variance = 0.15 ** 2  # Random variance in acceleration
    measurement_variance = np.array([[9, 0], [0, 9]])  # we will assume a constant measurement uncertainty
    # actual = np.r_[
    #     np.asarray([[-400 + 30 * t, 300] for t in np.arange(start=0, stop=15, step=1)]),
    #     np.asarray([[2 * math.pi * 300 * t, 300] for t in np.arange(start=15, stop=35, step=1)])
    # ]
    # print(actual.shape)
    measurement = np.array([
        [-393.66, 300.4], [- 375.93, 301.78], [- 351.04, 295.1], [- 328.96, 305.19], [- 299.35, 301.06],
        [- 273.36, 302.05], [- 245.89, 300], [- 222.58, 303.57], [- 198.03, 296.33], [- 174.17, 297.65],
        [- 146.32, 297.41], [- 123.72, 299.61], [- 103.47, 299.6], [- 78.23, 302.39], [- 52.63, 295.04],
        [- 23.34, 300.09], [25.96, 294.72], [49.72, 298.61], [76.94, 294.64], [95.38, 284.88], [119.83, 272.82],
        [144.01, 264.93], [161.84, 251.46], [180.56, 241.27], [201.42, 222.98], [222.62, 203.73], [239.4, 184.1],
        [252.51, 166.12], [266.26, 138.71], [271.75, 119.71], [277.4, 100.41], [294.12, 79.76], [301.23, 50.62],
        [291.8, 32.99], [299.89, 2.14]
    ])


    def transition_matrix(delta_t) -> np.ndarray:
        return np.array([
            [1, delta_t, 0.5 * delta_t ** 2, 0, 0, 0],
            [0, 1, delta_t, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, delta_t, 0.5 * delta_t ** 2],
            [0, 0, 0, 0, 1, delta_t],
            [0, 0, 0, 0, 0, 1]
        ])


    def estimate_uncertainty() -> np.ndarray:
        # Assuming uncertainty in X and Y direction are not correlated
        # Assuming it is constant
        return 500 * np.eye(6)


    def process_noise(time_change, a_variance) -> np.ndarray:
        p_noise = np.array([
            [time_change ** 4 / 4, time_change ** 3 / 2, time_change ** 2 / 2],
            [time_change ** 3 / 2, time_change ** 2, time_change],
            [time_change ** 2 / 2, time_change, 1]
        ])

        # Assuming Process noise in X and Y direction are not correlated and are the same
        return np.block([
            [p_noise, np.zeros((3, 3))],
            [np.zeros((3, 3)), p_noise]
        ]) * a_variance


    def dynamic_model(state: np.ndarray, control: np.ndarray, time_: float) -> np.ndarray:
        return transition_matrix(delta_time) @ state  # No control variable and is LTI


    def estimate_uncertainty_extrapolate(prev_estimate_uncertainty: np.ndarray) -> np.ndarray:
        # Assuming estimate errors in X and Y directions are not correlated
        return transition_matrix(delta_time) @ prev_estimate_uncertainty @ transition_matrix(
            delta_time).T + process_noise(delta_time, acceleration_variance)


    observation_matrix = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0]
    ])

    kalman_filter = MultiDimensionalKalmanFilter(
        dynamic_model=dynamic_model,
        covariance_model=estimate_uncertainty_extrapolate,
        observation_matrix=observation_matrix,
        init_state=np.array([0, 0, 0, 0, 0, 0]),
        init_estimate_uncertainty=estimate_uncertainty()
    )

    state_estimates = np.asarray([
        kalman_filter.next_state_estimate(
            measure,
            measurement_variance
        ) for measure in measurement
    ]).T

    time = np.arange(start=0, stop=len(measurement), step=1)
    go.Figure(
        data=[
            # go.Scatter(x=actual.T[0], y=actual.T[1], mode="lines+markers", name="Actual"),
            go.Scatter(x=measurement.T[0], y=measurement.T[1], mode="lines+markers", name="Measurement"),
            go.Scatter(x=state_estimates[0], y=state_estimates[3], mode="lines+markers", name="Estimate"),
        ],
        layout=go.Layout(showlegend=True, title="Vehicle Location Estimation in 2D", xaxis=dict(title="x - position"),
                         yaxis=dict(title="y - positions"))
    ).show()
