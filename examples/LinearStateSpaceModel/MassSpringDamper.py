import numpy as np
from Unconstrained.LinearStateSpaceModel import LinearStateSpaceModel

if __name__ == '__main__':
    m = 1  # Spring mass
    b = -0.5  # damper value set to cause instability
    k = 5  # Spring constant

    A = [[0, 1],
         [-k / m, -b / m]]

    B = [[0],
         [1 / m]]

    mass_spring = LinearStateSpaceModel(
        A=np.asarray(A),
        B=np.asarray(B),
        init_state=np.asarray([1, 1])
    )
    mass_spring.ode_gain(
        params=dict(control=np.asarray([[1] for _ in mass_spring.time]), gain=np.asarray([[2, 1]]))
    )

    mass_spring.plot_states()
    mass_spring.plot_controller()
    mass_spring.plot_output()
