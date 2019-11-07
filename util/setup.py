'''
@Author: Yu Di
@Date: 2019-10-27 19:17:46
@LastEditors: Yudi
@LastEditTime: 2019-10-28 11:13:06
@Company: Cardinal Operation
@Email: yudi@shanshu.ai
@Description: 
'''
# from distutils.core import setup
from setuptools import Extension, setup, find_packages, dist
from codecs import open
from os import path

dist.Distribution().fetch_build_eggs(['numpy>=1.11.2'])
try:
    import numpy as np
except ImportError:
    exit('Please install numpy>=1.11.2 first.')

try:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext
except ImportError:
    USE_CYTHON = False
else:
    USE_CYTHON = True

__version__ = '1.0.0'

ext = '.pyx' if USE_CYTHON else '.c'
cmdclass = {}

ext = '.pyx' if USE_CYTHON else '.c'

extensions = [
    Extension(
        name='matrix_factorization',
        sources=['matrix_factorization' + ext],
        include_dirs=[np.get_include()]),
    Extension(
        name='slim',
        sources=['slim' + ext],
        include_dirs=[np.get_include()]),
    Extension(
        name='similarities', 
        sources=['similarities' + ext], 
        include_dirs=[np.get_include()])
]

if USE_CYTHON:
    ext_modules = cythonize(extensions)
    cmdclass.update({'build_ext': build_ext})
else:
    ext_modules = extensions

setup(
    name='fair-comparison',
    author='Yu Di',
    ext_modules=ext_modules,
)

# from distutils.core import setup
# from Cython.Build import cythonize
# import numpy

# setup(
#     ext_modules = cythonize("slim.pyx"),
#     include_dirs = [numpy.get_include()]
# )

# # python setup.py build_ext --inplace