|pypi| |zenodo| |version| |python| |build| |ruff|

.. |pypi| image:: https://img.shields.io/pypi/v/physbeh.svg
    :target: https://pypi.org/project/physbeh/
    :alt: Pypi Package

.. |zenodo| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.15494995.svg
  :target: https://doi.org/10.5281/zenodo.15494995

.. |version| image:: https://img.shields.io/badge/version-0.6.4-orange.svg
    :target: https://github.com/FelipeCybis/physbeh/
    :alt: physbeh version

.. |python| image:: https://img.shields.io/badge/python-3.10_%7C_3.11_%7C_3.12_%7C_3.13-blue.svg
    :target: https://www.python.org/
    :alt: Python

.. |build| image:: https://github.com/FelipeCybis/physbeh/actions/workflows/run_workflows.yml/badge.svg
    :target: https://github.com/FelipeCybis/physbeh/actions/workflows/run_workflows.yml
    :alt: Build

.. |ruff| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
    :target: https://github.com/astral-sh/ruff
    :alt: Ruff

PhysBeh
=======

.. start-quickstart
.. start-about

This repository contains the main function and classes written in Python for
analysis of tracking data in the Physics for Medicine lab.

The code is written in the format of a Python package (`physbeh`).

This is the core package used for the behavioral analysis in the following paper:

1. Cybis Pereira, F., Castedo, S.H., Meur-Diebolt, S.L., Ialy-Radio, N., Bhattacharya, S., Ferrier, J., Osmanski, B.F., Cocco, S., Monasson, R., Pezet, S., et al. (2026). A vascular code for speed in the spatial navigation system. Cell Reports 45. https://doi.org/10.1016/j.celrep.2025.116791.

This package was implemented for easily manipulation of DeepLabCut output data.
Today there are more structured and well-maintained packages for this purpose, such as
[`movement`](https://github.com/neuroinformatics-unit/movement).


.. stop-about

Installation
------------

.. start-installation

`physbeh` can be installed from PyPI normally with `pip install physbeh`. Or you can
install it from source

1. Install PhysBeh from source
******************************

.. code-block:: bash

  python -m pip install git+https://github.com/FelipeCybis/physbeh.git

2. Check installation
*********************

Check that all tests pass:

.. code-block:: python

  import physbeh

If no error is raised, you have installed PhysBeh correctly.

.. stop-installation

.. stop-quickstart

Authors
-------

- Felipe Cybis Pereira (felipe.cybispereira@gmail.com)
