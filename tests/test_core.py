import unittest
import numpy as np
from sklearn.linear_model import Ridge
from src.core import metrics, ridge_path


class CoreTests(unittest.TestCase):
    def test_metric_identity_and_scale(self):
        y=np.array([1.,-2.,3.,4.]);p=np.array([.5,-1.,1.,2.])
        m=metrics(y,p)
        identity=(2*np.mean(y*p)-np.mean(p*p))/np.mean(y*y)
        self.assertAlmostEqual(m['r2'],identity)
        self.assertAlmostEqual(m['sharpe'],metrics(y,p*3)['sharpe'])
        self.assertAlmostEqual(m['sharpe_var'],m['sharpe']/np.sqrt(1-m['sharpe']**2))
        self.assertEqual(metrics(y,np.zeros(4))['sharpe'],0.)

    def test_ridge_scaling_and_minimum_norm(self):
        rng=np.random.default_rng(9)
        for n,p in [(20,7),(7,20),(7,7)]:
            x=rng.normal(size=(n,p));y=rng.normal(size=n)
            got=ridge_path(x,y,[0,.3])
            np.testing.assert_allclose(got[:,0],np.linalg.pinv(x)@y,atol=1e-10)
            expected=Ridge(alpha=n*.3,fit_intercept=False,solver='svd').fit(x,y).coef_
            np.testing.assert_allclose(got[:,1],expected,atol=1e-10)

    def test_rank_deficient_ridgeless(self):
        x=np.array([[1.,1.],[2.,2.],[3.,3.]])
        np.testing.assert_allclose(ridge_path(x,np.array([2.,4.,6.]),[0])[:,0],[1.,1.])


if __name__=='__main__':
    unittest.main()
