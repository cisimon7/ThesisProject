from unittest import TestCase
from systems import *


class TestRiccatiEquation(TestCase):

    def setUp(self) -> None:
        pass

    def test_system1(self):
        system = ConstraintRiccatiSystem(**system1)
        system.alpha = 1
        system.ext_u0 = False
        system.ode_solve(time_space=np.linspace(0, 10, int(2E3)))
        system.plot_states()
        # system.plot_output()

    def test_system2(self):
        system = ConstraintRiccatiSystem(**system2)
        system.alpha = 0.0001
        system.ext_u0 = False
        system.ode_solve(time_space=np.linspace(0, 20, int(2E3)))
        system.plot_output()

    def test_system3(self):
        system = ConstraintRiccatiSystem(**system3)
        system.alpha = 1
        system.ext_u0 = False
        system.ode_solve(time_space=np.linspace(0, 10, int(2E3)))
        system.plot_output()

    def test_system5(self):
        system = ConstraintRiccatiSystem(**system5)
        system.alpha = 0.0001
        system.ext_u0 = False
        system.ode_solve(time_space=np.linspace(0, 15, int(2E3)))
        # system.plot_states()
        system.plot_output()
        # system.plot_controller()

    def test_system6(self):
        system = ConstraintRiccatiSystem(**system6)
        system.alpha = 0.001
        system.ext_u0 = False
        system.ode_solve(time_space=np.linspace(0, 5, int(2E3)))
        system.plot_states()
        system.plot_output()
        system.plot_controller()

    def test_system7(self):
        system = ConstraintRiccatiSystem(**system7)
        system.alpha = 1
        system.ext_u0 = True
        system.ode_solve(time_space=np.linspace(0, 5, int(2E3)))
        # system.plot_states()
        # system.plot_output()
        # system.plot_controller()

    def test_random_system(self):
        system = ConstraintRiccatiSystem(**random_system(n_size=6, u_size=5, k_size=4))
        system.alpha = 1
        system.ext_u0 = True
        system.ode_solve(time_space=np.linspace(0, 10, int(2E3)))
        system.plot_states()
        system.plot_output()
        system.plot_controller()
