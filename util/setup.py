'''
@Author: Yu Di
@Date: 2019-10-27 19:17:46
@LastEditors: Yudi
@LastEditTime: 2019-10-28 11:13:06
@Company: Cardinal Operation
@Email: yudi@shanshu.ai
@Description: 
'''
from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize("slim.pyx"),
    include_dirs = [numpy.get_include()]
)

# python setup.py build_ext --inplace