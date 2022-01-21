import numpy as np
import plotly.graph_objects as go

from KalmanFilter import KalmanFilter

if __name__ == "__main__":
    measurement_error = 5
    measurements = np.array([48.54, 47.11, 55.01, 55.15, 49.89, 40.85, 46.72, 50.05, 51.27, 49.95])
    actual_height = np.asarray([50 for _ in measurements])  # Actual height is always constant

    kalman_filter = KalmanFilter(
        system_dynamic_model_func=(lambda x: x),
        variance_extrapolation_func=(lambda x: x),
        init_state=np.array([60]),
        init_variance=np.array([255])
    )

    estimates = np.asarray([
        kalman_filter.next_state_estimate(
            measurement=np.array([measure]),
            measurement_uncertainty=np.array([25])
        )[0] for measure in measurements
    ])

    x_axis = np.arange(start=0, stop=len(measurements), step=1)
    go.Figure(
        data=[
            go.Scatter(x=x_axis, y=actual_height, name="True Value"),
            go.Scatter(x=x_axis, y=measurements, name="Measurements"),
            go.Scatter(x=x_axis, y=estimates.flatten(), name="Estimate"),
        ]
    ).show()
