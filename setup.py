import numpy
import os, sys, os.path
from distutils.core import setup

setup(
    name='pyritz',
    version='0.0.1',
    url='https://github.com/lukastk/PyRitz',
    author='Lukas Kikuchi, Rajesh Singh, Mike Cates, Ronojoy Adhikari',
    author_email='ltk26@cam.ac.uk, rs2004@cam.ac.uk, m.e.cates@damtp.cam.ac.uk, ra413@cam.ac.uk',
    license='MIT',
    description='A python package for direct variational minimisation, specifically suited for finding Freidlin-Wentzell instantons.',
    platforms='Tested on Linux and Mac',
    libraries=[],
    packages=['pyritz', 'pyritz.interpolation'],
)
