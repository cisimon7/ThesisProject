from systems import *
from unittest import TestCase
from Constrained.CompactSystem import CompactSystem


class TestIdea(TestCase):

    def test_sys(self):
        sys = CompactSystem(**system5)
        sys.ode_solve()
        sys.plot_states()
        # sys.plot_output()
        # sys.plot_controller()
        # sys.plot_overview()
