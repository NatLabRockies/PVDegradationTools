.. _tutorials:

==========
Tutorials
==========

PVDeg provides comprehensive tutorials organized by topic. Choose your preferred environment:

Jupyter Book (Recommended)
---------------------------

Interactive tutorials with live execution: `PVDeg Jupyter Book <https://nrel.github.io/PVDegradationTools/intro.html>`_

- Click the 🚀 rocket icon to launch notebooks in `Google Colab <https://colab.research.google.com/>`_
- **Development Preview:** See latest changes at `dev-preview <https://natlabrockies.github.io/PVDegradationTools/dev-preview/intro.html>`_

Binder
------

Run tutorials in your browser without installation:

.. image:: https://mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/gh/NREL/PVDegradationTools/main
    :alt: Binder

Local Installation
------------------

1. **Install PVDeg** (see :ref:`installation`)

2. **Clone the repository** to access tutorial notebooks:

   .. code-block:: bash

      git clone https://github.com/NREL/PVDegradationTools.git
      cd PVDegradationTools

3. **Start Jupyter:**

   .. code-block:: bash

      jupyter notebook

4. **Navigate to tutorials** organized by category:

   - ``01_basics/`` - Introduction to PVDeg fundamentals
   - ``02_degradation/`` - Degradation mechanism modeling
   - ``03_monte_carlo/`` - Monte Carlo uncertainty analysis
   - ``04_geospatial/`` - Geospatial and HPC scenarios
   - ``05_advanced/`` - Advanced topics and API access
   - ``10_workshop_demos/`` - Workshop demonstrations
   - ``tools/`` - Standalone analysis tools

NREL HPC (Kestrel)
------------------

Running notebooks on Kestrel is documented on the `NREL HPC Documentation <https://natlabrockies.github.io/HPC/Documentation/Development/Jupyter/>`_.

**Important:** Register a custom iPykernel before running notebooks on Kestrel:

.. code-block:: bash

   python -m ipykernel install --user --name=pvdeg-env

Replace ``pvdeg-env`` with your conda environment name. Restart your Jupyter server to load the new kernel, which will appear in the kernel selection dropdown.

For more information on validating notebook outputs and best practices, see the :ref:`contributing` guide.
