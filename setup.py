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
          'click',
          'matplotlib',
          'multiset',
          'netabc',
          'networkx',
          'numba',
          'pandas',
          'scipy'
      ],
      python_requires='>=3.1.*',
      entry_points={
          'console_scripts': [
              'stamford_graph = stamford.graph:command',
          ],
          'netkappa_functions': [
              'sar = stamford.network:sar',
              'emsar = stamford.network:emsar'
          ],
          'netabc_commands': [
              'stamford  = stamford.network:command',
              'plot_stamford_data = stamford.plot:plot_stamford_data',
              'plot_stamford_demo = stamford.plot:plot_stamford_demo',
              'plot_stamford_act = stamford.plot:plot_stamford_act',
              'plot_stamford_cens = stamford.plot:plot_stamford_cens',
              'plot_stamford_intro = stamford.plot:plot_stamford_intro',
              'plot_stamford_wass = stamford.plot:plot_stamford_wass',
              'plot_scaled_activity = stamford.plot:plot_scaled_activity',
              'write_stamford_act = stamford.plot:write_stamford_act',
              'write_stamford_dist = stamford.plot:write_stamford_dist',
              'write_stamford_intro = stamford.plot:write_stamford_intro',
              'write_stamford_multi = stamford.plot:write_stamford_multi',
              'write_stamford_net = stamford.plot:write_stamford_net',
              'write_stamford_wass = stamford.plot:write_stamford_wass',
          ],
      },
)
