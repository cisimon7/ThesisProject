import numpy as np
from LinearConstraintStateSpaceModel import LinearConstraintStateSpaceModel

if __name__ == '__main__':
    system = LinearConstraintStateSpaceModel(
        A=np.eye(4),
        B=np.array([
            [0, 1 / 67, 0, 1 / 67],
            [0, -(4 / 67), 0, -(4 / 67)],
            [0, 7 / 67, 0, 7 / 67],
            [0, 1 / 67, 0, 1 / 67]
        ]),
        G=np.array([[1, 0, -1, 1], [0, 1, 1, 1], [1, 1, 0, 2], [2, 3, 1, 5]]),
        F=np.eye(4, 6),
        init_state=1000*np.array([1, 1, 1, 1])
    )
    system.ode_gain_solve(time_space=np.linspace(0, 20, int(2E3)))
    system.plot_states()
