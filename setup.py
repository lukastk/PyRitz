import numpy
import os, sys, os.path
from distutils.core import setup


setup(
    name='pyritz',
    version='0.0.1',
    url='https://github.com/',
    author='Rajesh Singh, R. Adhikari and Lukas Kikuchi',
    author_email='rs2004@cam.ac.uk, ra413@cam.ac.uk and ltk26@cam.ac.uk',
    license='MIT',
    description='Python library for computing Instanton paths',
    platforms='Tested on Linux and Mac',
    libraries=[],
    packages=['pyritz', 'pyritz.expansion'],
)
