import unittest
import numpy as np
from ControlSystem import ControlSystem


# Testing class using a simple Pendulum System
class TestControlSystem(unittest.TestCase):

    def setUp(self) -> None:
        self.system = ControlSystem(A=np.eye(4), B=np.eye(4))

    # Assert shapes of input
    def test_lqr_controller(self):
        self.system.gain_lqr()
