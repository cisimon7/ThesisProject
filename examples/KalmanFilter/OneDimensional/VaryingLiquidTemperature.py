import numpy as np
import plotly.graph_objects as go
from OneDimensionalKalmanFilter import OneDimensionalKalmanFilter

if __name__ == "__main__":
    """
    ESTIMATING THE TEMPERATURE OF A HEATING LIQUID 
    In this case, though, the system dynamics is not constant - the liquid is heating at a rate of 0.1oC every second.
    However, we assume not to know this, which means our process is not well-defined. 
    This is shown by the high process_noise_variance
    The measurements are taken every 5 seconds. The system dynamics is constant.
    
    Kalman filter would take into account the fact that model is not accurate
    """

    measurement_error = 0.1
    process_noise_variance = 0.15

    # set of 10 measurements
    measurements = np.array([50.45, 50.967, 51.6, 52.106, 52.492, 52.819, 53.433, 54.007, 54.523, 54.99])
    actual_temperature = np.asarray([50.479, 51.025, 51.5, 52.003, 52.494, 53.002, 53.499, 54.006, 54.498, 54.991])
    time = np.array([5 * i for i in np.arange(start=0, stop=len(measurements), step=1.0)])

    k_filter = OneDimensionalKalmanFilter(
        system_dynamic_model_func=(lambda x: x),  # Actual model should be (lambda x: x + 0.5)
        variance_extrapolation_func=(lambda x: x + process_noise_variance),
        init_state=np.array([10]),
        init_variance=np.array([10_000])
    )

    estimates = np.asarray([
        k_filter.next_state_estimate(
            measurement=np.array([measure]),
            measurement_uncertainty=np.array([measurement_error ** 2])
        )[0] for measure in measurements
    ])

    go.Figure(
        data=[
            go.Scatter(x=time, y=actual_temperature, name="True Value"),
            go.Scatter(x=time, y=measurements, name="Measurements"),
            go.Scatter(x=time, y=estimates.flatten(), name="Estimate"),
        ],
        layout=go.Layout(
            title="Varying Liquid Temperature Estimation",
            yaxis=dict(title="Temperature (celsius)"),
            xaxis=dict(title="Time")
        )
    ).show()

    k_filter.plot_kalman_gain()
    k_filter.plot_uncertainties()
