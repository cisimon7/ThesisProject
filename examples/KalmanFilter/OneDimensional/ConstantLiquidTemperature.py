import numpy as np
import plotly.graph_objects as go
from KalmanFilter.OneDimensionalKalmanFilter import OneDimensionalKalmanFilter

if __name__ == "__main__":
    """
    ESTIMATING THE TEMPERATURE OF THE LIQUID IN A TANK
    We assume that at steady state the liquid temperature is constant. 
    However, some fluctuations in the true liquid temperature are possible. 
    We can describe the system dynamics by the following equation: xn=T+wn
        where:
            T is the constant temperature
            wn is a random process noise with variance q
    The measurements are taken every 5 seconds.
    """

    measurement_error = 0.1
    process_noise_variance = 0.0001

    # set of 10 measurements
    measurements = np.array([49.95, 49.967, 50.1, 50.106, 49.992, 49.819, 49.933, 50.007, 50.023, 49.99])
    actual_temperature = np.asarray([49.979, 50.025, 50, 50.003, 49.994, 50.002, 49.999, 50.006, 49.998, 49.991])
    time = np.array([5 * i for i in np.arange(start=0, stop=len(measurements), step=1)])

    kalman_filter = OneDimensionalKalmanFilter(
        system_dynamic_model_func=(lambda x: x),
        variance_extrapolation_func=(lambda x: x + process_noise_variance),
        init_state=np.array([10]),
        init_variance=np.array([10_000])
    )

    estimates = np.asarray([
        kalman_filter.next_state_estimate(
            measurement=np.array([measure]),
            measurement_uncertainty=np.array([measurement_error ** 2])
        )[0] for measure in measurements
    ])

    # Plotting results
    go.Figure(
        data=[
            go.Scatter(x=time, y=actual_temperature, name="True Value"),
            go.Scatter(x=time, y=measurements, name="Measurements"),
            go.Scatter(x=time, y=estimates.flatten(), name="Estimate"),
        ],
        layout=go.Layout(
            title="Liquid Temperature Estimation",
            yaxis=dict(title="Temperature (celsius)"),
            xaxis=dict(title="Measurement number")
        )
    ).show()

    kalman_filter.plot_kalman_gain()
    kalman_filter.plot_uncertainties()
