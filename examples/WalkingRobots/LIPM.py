import numpy as np

# 3D Linear Inverted Pendulum Model: A Simple Modeling for Walking Bipedal Robots
# CoM: Centre of Mass

from Unconstrained.LinearStateSpaceModel import LinearStateSpaceModel

if __name__ == '__main__':
    m = 1  # mass
    g = 9.81  # Gravity constant
    zc = 1  # Constant height plane for CoM

    A = np.array([
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [g / zc, 0, 0, 0, 0, 0],
        [0, g / zc, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ])

    B = np.diag([0, 0, 0, 1 / (m * zc), 1 / (m * zc), 0])

    C = np.block([
        [np.eye(3), np.zeros((3, 3))]
    ])

    xdot = LinearStateSpaceModel(A, 0 * B, C)
    xdot.time = np.linspace(0, 5, int(2E2))
    xdot.ode_gain(
        params=dict(control=np.asarray([np.r_[0, 0, 0, 0, 0, 0] for _ in xdot.time]), gain=None),
        init_state=np.r_[0, 0, zc, 0.5, 0.7, 0]
    )
    xdot.plot_output()
