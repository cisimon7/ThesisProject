from unittest import TestCase
from tests.test_systems import *


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
        system.plot_states()

    def test_system2(self):
        system = constraint_system2
        system.ode_gain_solve(time_space=np.linspace(0, 20, int(2E3)))
        system.plot_states()

    def test_system4(self):
        system = constraint_system5
        system.ode_gain_solve(time_space=np.linspace(0, 15, int(2E3)))
        system.plot_states()
        system.plot_controller()
        system.plot_output()

    def test_random_system(self):
        system = random_system(n_size=5, u_size=3, k_size=4)
        system.ode_gain_solve(time_space=np.linspace(0, 20, int(2E3)))
        system.plot_states()
        system.plot_controller()
        system.plot_output()
