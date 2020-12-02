#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='stamford',
      version='0.1',
      description='Processing scripts and model front-ends for Stamford Hill COVID-19 Survey Project',
      author=['William Waites'],
      author_email='william.waites@lshtm.ac.uk',
      keywords=['COVID-19'],
      classifiers=[
          # How mature is this project? Common values are
          #   3 - Alpha
          #   4 - Beta
          #   5 - Production/Stable
          'Development Status :: 4 - Beta',

          # Intended audience
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',

          # License
          'License :: OSI Approved :: GNU General Public License (GPL)',
          # Specify the Python versions you support here. In particular,
          # ensure that you indicate whether you support Python 2, Python 3
          # or both.
          'Programming Language :: Python :: 3',
      ],
      license='GPLv3',
      packages=find_packages(),
      install_requires=[
          'matplotlib',
          'multiset',
          'networkx',
          'numba',
          'pandas',
          'tqdm',
          'scipy'
      ],
      python_requires='>=3.1.*',
      entry_points={
          'console_scripts': [
              'stamford_graph = stamford.graph:command',
              'stamford_house = stamford.house:command',
              'stamford_plot = stamford.plot:command',
          ],
      },
)
