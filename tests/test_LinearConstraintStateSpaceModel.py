from unittest import TestCase
from tests.systems import *


class TestLinearConstraintStateSpaceModel(TestCase):

    def setUp(self) -> None:
        pass

    def test_zdot(self):
        system = constraint_system2
        dot = system.z_dot_gain(state=system.N @ system.init_state, time=0, gain=system.gain_lqr())

        state = (system.N.T @ dot)
        print(state)

    def test_system1(self):
        system = constraint_system1
        system.ode_gain_solve(time_space=np.linspace(0, 10, int(2E3)))
        # system.plot_states()
        system.plot_output()

    def test_system2(self):
        system = constraint_system2
        system.ode_gain_solve(time_space=np.linspace(0, 20, int(2E3)))
        system.plot_output()

    def test_system3(self):
        system = constraint_system3
        system.ode_gain_solve(time_space=np.linspace(0, 10, int(2E3)))
        system.plot_output()

    def test_system5(self):
        system = constraint_system5
        system.ode_gain_solve(time_space=np.linspace(0, 15, int(2E3)))
        # system.plot_states()
        # system.plot_controller()
        system.plot_output()

    def test_system6(self):
        system = constraint_system6
        system.ode_gain_solve(time_space=np.linspace(0, 5, int(2E3)))
        # system.plot_states()
        # system.plot_controller()
        system.plot_output()

    def test_system7(self):
        system = constraint_system7
        system.ode_gain_solve(time_space=np.linspace(0, 5, int(2E3)))
        # system.plot_states()
        # system.plot_controller()
        system.plot_output()

    def test_random_system(self):
        system = random_system(n_size=6, u_size=5, k_size=4)
        system.ode_gain_solve(time_space=np.linspace(0, 10, int(2E3)))
        # system.plot_states()
        # system.plot_controller()
        system.plot_output()
