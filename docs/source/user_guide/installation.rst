.. _installation:

Installation
============

PVDeg releases may be installed using pip. Compatible with Python 3.10 and above.

For a quick start, see the **Installation** section in our `README.md <https://github.com/NREL/PVDegradationTools/blob/main/README.md#installation>`_.

Base Install
------------

To install PVDeg with core functionality:

.. code-block:: bash

    pip install pvdeg

This installs PVDeg with all required dependencies for basic degradation modeling, including:

* Core scientific computing libraries (numpy, pandas, scipy)
* PV modeling with pvlib
* Weather data access (NREL-rex)
* Geospatial tools (cartopy, geopy)
* Jupyter notebook support (jupyterlab, notebook)
* Pre-commit hooks for development

Optional Dependencies
---------------------

PVDeg offers optional dependency groups for specific use cases. You can install these using the bracket syntax:

.. code-block:: bash

    pip install pvdeg[group_name]

Available Optional Groups
~~~~~~~~~~~~~~~~~~~~~~~~~

**sam** - System Advisor Model Integration
    Adds NREL-PySAM for detailed system performance modeling and integration with SAM.
    Required for agrivoltaics and advanced system modeling workflows.

    .. code-block:: bash

        pip install pvdeg[sam]

**docs** - Documentation Building
    Installs tools for building Sphinx documentation including:

    * sphinx and themes
    * nbsphinx for notebook integration
    * sphinx-gallery for example galleries

    .. code-block:: bash

        pip install pvdeg[docs]

**test** - Testing and Validation
    Includes pytest, pytest-cov, nbval for notebook validation, and scikit-learn for testing.
    Essential for contributors running the test suite.

    .. code-block:: bash

        pip install pvdeg[test]

**books** - Jupyter Book Publishing
    Adds jupyter-book for building and publishing tutorial documentation.

    .. code-block:: bash

        pip install pvdeg[books]

**all** - Everything
    Installs all optional dependencies above. Recommended for developers.

    .. code-block:: bash

        pip install pvdeg[all]

Using Conda Environments
-------------------------

While PVDeg is installed via pip, you can use conda to manage your Python environment:

1. Create a new conda environment:

   .. code-block:: bash

       conda create -n pvdeg python=3.11
       conda activate pvdeg

2. Install PVDeg with pip:

   .. code-block:: bash

       pip install pvdeg[all]

3. Register the environment as a Jupyter kernel (important for HPC systems):

   .. code-block:: bash

       python -m ipykernel install --user --name=pvdeg

   This allows you to select the ``pvdeg`` kernel when running Jupyter notebooks,
   especially important on HPC systems like NREL's Kestrel.

Developer Installation
----------------------

If you want to contribute to PVDeg or modify the source code, install in editable mode:

1. Fork the repository on GitHub and clone your fork:

   .. code-block:: bash

       git clone https://github.com/YOUR-USERNAME/PVDegradationTools.git
       cd PVDegradationTools

2. Create and activate a virtual environment:

   Using venv (Python standard library):

   .. code-block:: bash

       python -m venv pvdeg-dev
       source pvdeg-dev/bin/activate  # On Windows: pvdeg-dev\Scripts\activate

   Or using conda:

   .. code-block:: bash

       conda create -n pvdeg-dev python=3.11
       conda activate pvdeg-dev

3. Install in editable mode with all dependencies:

   .. code-block:: bash

       pip install -e .[all]

   The ``-e`` flag installs the package in "editable" mode, meaning changes you make
   to the source code are immediately reflected without reinstalling.

4. Install pre-commit hooks for code quality checks:

   .. code-block:: bash

       pre-commit install

   This will automatically run code formatting (black) and linting (flake8) before each commit.

5. (Optional but recommended for HPC) Register the kernel for Jupyter:

   .. code-block:: bash

       python -m ipykernel install --user --name=pvdeg-dev

6. Verify your installation by running the test suite:

   .. code-block:: bash

       pytest pvdeg

For complete developer guidelines including contribution workflow, code style,
and testing requirements, see :ref:`contributing`.

Troubleshooting
---------------

**Import errors after installation**
    Make sure you're in the correct environment and that the installation completed successfully.
    Try ``pip list | grep pvdeg`` to verify the package is installed.

**HPC/Kestrel Jupyter kernel issues**
    If your environment doesn't appear in Jupyter on HPC systems, ensure you've run:
    ``python -m ipykernel install --user --name=your-env-name``

**Dependency conflicts**
    Use a fresh virtual environment if you encounter version conflicts with existing packages.

**Cartopy installation issues**
    Cartopy requires GEOS and PROJ libraries. On some systems, you may need to install these
    via conda: ``conda install -c conda-forge cartopy``

Version Compatibility
---------------------

* **Python**: 3.10, 3.11, 3.12, 3.13
* **Operating Systems**: Linux, macOS, Windows
* **Key Dependencies**:
    * pvlib >= 0.12.0
    * numpy >= 1.19.3
    * pandas (compatible versions)
    * h5py <= 3.14.0 (pinned due to compatibility issues)

For the most up-to-date compatibility information, see the ``pyproject.toml`` file in the repository.
