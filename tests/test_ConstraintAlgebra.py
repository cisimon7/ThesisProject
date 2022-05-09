from unittest import TestCase

from Constrained.ConstraintAlgebra import ConstraintAlgebra
from tests.systems import *


class TestConstraintAlgebra(TestCase):

    def setUp(self) -> None:
        pass

    def test_system1(self):
        system = ConstraintAlgebra(**system1)
        system.ode_gain(time_space=np.linspace(0, 10, int(2E3)))
        # system.plot_states()
        system.plot_output()

    def test_system2(self):
        system = ConstraintAlgebra(**system2)
        system.ode_gain(time_space=np.linspace(0, 20, int(2E3)))
        system.plot_output()

    def test_system3(self):
        system = ConstraintAlgebra(**system3)
        system.ode_gain(time_space=np.linspace(0, 10, int(2E3)))
        system.plot_output()

    def test_system5(self):
        system = ConstraintAlgebra(**system5)
        system.ode_gain(time_space=np.linspace(0, 15, int(2E3)))
        # system.plot_states()
        # system.plot_controller()
        system.plot_output()

    def test_system6(self):
        system = ConstraintAlgebra(**system6)
        system.ode_gain(time_space=np.linspace(0, 5, int(2E3)))
        # system.plot_states()
        system.plot_controller()
        system.plot_output()

    def test_system7(self):
        system = ConstraintAlgebra(**system7)
        system.ode_gain(time_space=np.linspace(0, 5, int(2E3)))
        # system.plot_states()
        # system.plot_controller()
        system.plot_output()

    def test_random_system(self):
        system = random_system(n_size=6, u_size=5, k_size=4)
        system.ode_gain(time_space=np.linspace(0, 10, int(2E3)))
        # system.plot_states()
        # system.plot_controller()
        system.plot_output()
