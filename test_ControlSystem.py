import unittest
import numpy as np
from ControlSystem import ControlSystem


class TestControlSystem(unittest.TestCase):

    def setUp(self) -> None:
        self.system = ControlSystem(A=np.eye(3), B=np.eye(3))

    # Assert shapes of input
