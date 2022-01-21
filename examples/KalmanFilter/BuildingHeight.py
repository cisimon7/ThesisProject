import numpy as np
import plotly.graph_objects as go

from OneDimensionalKalmanFilter import OneDimensionalKalmanFilter

if __name__ == "__main__":
    """
    ESTIMATING THE HEIGHT OF A BUILDING
    Building height is constant with true height being 50 meters
    The altimeter measurement error (standard deviation) is 5 meters.
    Initial estimate is given by human approximation as 60  meters and human estimation error taken as 15 meters
    """

    measurement_error = 5

    # set of 10 measurements
    measurements = np.array([48.54, 47.11, 55.01, 55.15, 49.89, 40.85, 46.72, 50.05, 51.27, 49.95])
    actual_height = np.asarray([50 for _ in measurements])  # Actual height is always constant

    kalman_filter = OneDimensionalKalmanFilter(
        system_dynamic_model_func=(lambda x: x),
        variance_extrapolation_func=(lambda x: x),
        init_state=np.array([60]),
        init_variance=np.array([255])
    )

    estimates = np.asarray([
        kalman_filter.next_state_estimate(
            measurement=np.array([measure]),
            measurement_uncertainty=np.array([measurement_error ** 2])
        )[0] for measure in measurements
    ])

    x_axis = np.arange(start=0, stop=len(measurements), step=1)
    go.Figure(
        data=[
            go.Scatter(x=x_axis, y=actual_height, name="True Value"),
            go.Scatter(x=x_axis, y=measurements, name="Measurements"),
            go.Scatter(x=x_axis, y=estimates.flatten(), name="Estimate"),
        ],
        layout=go.Layout(
            title="Building Height Estimation",
            yaxis=dict(title="Height (meters)"),
            xaxis=dict(title="Measurement number")
        )
    ).show()

    kalman_filter.plot_kalman_gain()
    kalman_filter.plot_estimate_uncertainty()
