import unittest
import numpy as np
from analysis import *

class Tests(unittest.TestCase):
    def test_noiseless(self):
        t=np.linspace(0,180,100);p=fit_response(t,response(t,22,15,35));np.testing.assert_allclose(p,[22,15,35],atol=1e-6)
    def test_heldout_prediction(self):
        t=np.linspace(0,180,361);truth=response(t,22,15,35)
        for seed in [1,42,93]:
            y=truth+np.random.default_rng(seed).normal(0,.15,len(t));p=fit_response(t[t<=110],y[t<=110]);self.assertLess(np.sqrt(np.mean((response(t[t>110],*p)-truth[t>110])**2)),.15)
    def test_settling_formula(self):self.assertAlmostEqual(response(-35*np.log(.05),22,15,35),22+.95*15)
    def test_flat_data(self):
        with self.assertRaises(ValueError):fit_response(np.arange(20),np.ones(20))
