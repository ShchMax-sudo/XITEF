import pybind11
from distutils.core import setup, Extension

ext_modules = [
    Extension(
        'sifting',
        ['Sifting.cpp', 'main.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=['-std=c++11'],
    ),
]

setup(
    name='sifting',
    version='0.0.1',
    author='ShchMax',
    ext_modules=ext_modules,
    requires=['pybind11']
)