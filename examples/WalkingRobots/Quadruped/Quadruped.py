import numpy as np
from scipy.io import loadmat

from Constrained.ConstraintRiccati import ConstraintRiccatiSystem

if __name__ == '__main__':
    data = loadmat("quadruped.mat")
    A, B, G, x_init, x_desired = data['A'], data['B'], data['G'], data['x'], data['dx']

    system = ConstraintRiccatiSystem(
        A=np.asarray(A),
        B=np.asarray(B),
        G=np.asarray(G),
        init_state=np.asarray(x_init).flatten(), x_desired=np.asarray(x_desired).flatten()
    )
    print(x_desired[15:18])
    system.alpha = 0.0001
    system.ext_u0 = True
    system.ode_solve(time_space=np.linspace(0, 5, int(2E3)))
    system.plot_states(state_name="q")
    system.plot_x_states(interest=[15, 18])
    # system.plot_d_states()
    # system.plot_z_output()
    # system.plot_output()
    # system.plot_controller()
    # system.plot_overview()
    print("Done")
