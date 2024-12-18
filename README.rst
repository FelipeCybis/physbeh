PhysBeh
=======

.. start-quickstart

This repository contains the main function and classes written in Python for
analysis of tracking data in the Physics for Medicine lab.

The code is written in the format of a Python package (`physbeh`).

Installation
------------

.. start-installation

1. Setup a virtual environment
******************************

Create and activate a new python environment (if familiarisation with virtual
environments is needed, you can start `here
<https://docs.python.org/3/library/venv.html>`__ or `here
<https://ioflood.com/blog/python-venv-virtual-environment/>`__):

- Using `venv` (recommended):

  On Linux

  .. code-block:: bash

    python3 -m venv /path_to_env
    source /path_to_env/bin/activate

  On Windows

  .. code-block:: powershell

    python3 -m venv /path_to_env
    /path_to_env/Script/activate

- Using `conda <https://docs.conda.io/projects/conda/en/stable/>`_:

  .. code-block:: bash

    conda create -n physbeh python=3.11
    conda activate physbeh

2. Install PhysBeh from source
******************************

PhysBeh is a private package developed by Iconeus and Physics for Medicine and is not
available from PyPI.

.. code-block:: bash

  python -m pip install git+https://github.com/FelipeCybis/physbeh.git

3. Check installation
*********************

Check that all tests pass:

.. code-block:: python

  import physbeh

If no error is raised, you have installed PyfUS correctly.

.. stop-installation

.. stop-quickstart

Authors
-------

- Felipe Cybis Pereira (felipe.cybispereira@gmail.com)
