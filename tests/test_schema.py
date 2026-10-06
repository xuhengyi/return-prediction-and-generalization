import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from src.core import read_pair


class SchemaTests(unittest.TestCase):
    def test_test_columns_are_aligned_from_training(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            pd.DataFrame({'t':[1,2],'feature1':[10,20],'feature2':[3,4],'return':[.1,.2]}).to_csv(p/'pairA_train.csv',index=False)
            pd.DataFrame({'t':[3,4],'feature2':[7,8],'feature1':[30,40]}).to_csv(p/'pairA_test_features.csv',index=False)
            _,y,xt,t,names=read_pair(p,'A')
            np.testing.assert_array_equal(xt,[[30,7],[40,8]])
            np.testing.assert_array_equal(y,[.1,.2]) # No additional time shift.
            np.testing.assert_array_equal(t,[3,4])
            self.assertEqual(names,['feature1','feature2'])
            extra=pd.read_csv(p/'pairA_test_features.csv');extra['feature3']=0
            extra.to_csv(p/'pairA_test_features.csv',index=False)
            with self.assertRaises(ValueError):read_pair(p,'A')

    def test_reject_duplicate_times_and_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            pd.DataFrame({'t':[1,1],'feature1':[1,2],'return':[3,4]}).to_csv(p/'pairA_train.csv',index=False)
            pd.DataFrame({'t':[3,4],'feature1':[5,6]}).to_csv(p/'pairA_test_features.csv',index=False)
            with self.assertRaises(ValueError):read_pair(p,'A')
            pd.DataFrame({'t':[1,2],'feature1':[1,np.inf],'return':[3,4]}).to_csv(p/'pairA_train.csv',index=False)
            with self.assertRaises(ValueError):read_pair(p,'A')


if __name__=='__main__':unittest.main()
