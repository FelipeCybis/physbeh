# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['tracking_physmed',
 'tracking_physmed.gui',
 'tracking_physmed.tracking',
 'tracking_physmed.utils']

package_data = \
{'': ['*']}

install_requires = \
['matplotlib>=3.5.1,<4.0.0',
 'numpy>=1.22.1,<2.0.0',
 'opencv-python-headless>=4.5.5,<5.0.0',
 'pandas>=1.3.5,<2.0.0',
 'ruamel.yaml>=0.17.20,<0.18.0',
 'scipy>=1.7.3,<2.0.0',
 'tables>=3.7.0,<4.0.0']

setup_kwargs = {
    'name': 'tracking-physmed',
    'version': '0.1.0',
    'description': 'Miscellaneous functions for tracking analysis in Python using DeepLabCut at Physics for Medicine',
    'long_description': None,
    'author': 'Felipe Cybis Pereira',
    'author_email': 'felipe.cybis-pereira@espci.psl.eu',
    'maintainer': None,
    'maintainer_email': None,
    'url': None,
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.8,<3.11',
}


setup(**setup_kwargs)

