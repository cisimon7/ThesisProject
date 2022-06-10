import numpy as np
from scipy.io import loadmat

from Constrained.ConstraintRiccati import ConstraintRiccatiSystem

if __name__ == '__main__':
    data = loadmat("data.mat")
    A, B, C, G, g, tol, R_type, R_custom, ControllerSettings, ObserverSettings, x_desired, dx_desired = \
        data['System'][0][0].T

    system = ConstraintRiccatiSystem(A=A, B=B, C=C, G=G, g=g,
                                     init_state=(x_desired + 0.2*np.random.random(x_desired.shape)).flatten(),
                                     x_desired=x_desired, dx_desired=dx_desired)
    system.alpha = 0.0001
    system.ext_u0 = True
    system.ode_solve(time_space=np.linspace(0, 2.5, int(2E3)))
    system.plot_states(state_name="q")
    # system.plot_x_states()
    # system.plot_d_states()
    # system.plot_z_output()
    system.plot_output()
    system.plot_controller()
    # system.plot_overview()
