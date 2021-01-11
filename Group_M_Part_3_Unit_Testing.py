
import unittest
from unittest import mock
import os
os.chdir('C:\\Users\\kelennon\\Desktop\MSDS\\Summer 2019\\Wine Project')
from Wine import *

class test_function(unittest.TestCase): # inherit from unittest.TestCase

    def test_value(self): # test if value is almost equal to 0 
        self.assertAlmostEqual(value_calc(df)['value'].mean(),0)

    def test_mean_normed_rating(self): # test of points have been normalized
        self.assertAlmostEqual(value_calc(df)['points'].mean(),0)
          
if __name__ == '__main__':
    unittest.main()
