# PVDeg - Tutorials and Tools

PVDeg is an open-source Python package for modeling photovoltaic (PV) degradation. These tutorials demonstrate how to use PVDeg's modular functions, materials databases, and workflows for simulating degradation mechanisms (e.g., LeTID, hydrolysis, UV exposure) using weather data from NSRDB and PVGIS.

## Tutorial Categories

- `01_basics/` - Introduction to PVDeg fundamentals
- `02_degradation/` - Degradation mechanism modeling (LID, LETID, Van't Hoff)
- `03_monte_carlo/` - Monte Carlo uncertainty analysis
- `04_scenario/` - Scenario-based workflows and HPC analysis
- `05_geospatial/` - Geospatial analysis and world map visualizations
- `06_advanced/` - Advanced topics and API access
- `10_workshop_demos/` - Workshop and demonstration notebooks

For standalone analysis tools, see the [Tools](../tools/) directory.

## Running the Tutorials

### Jupyter Book

For learning to use PVDeg through our tutorials, or running the available tools online, see our [jupyter-book](https://NatLabRockies.github.io/PVDegradationTools/intro.html)
Clicking on the rocket-icon on the top allows you to launch the notebooks on [Google Colaboratory](https://colab.research.google.com/) for interactive mode.
Just uncomment the first line `pip install ...` to install the environment on each notebook if you follow this mode.

### Binder

To run these tutorials or tools in Binder, you can click here:
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/NatLabRockies/PVDegradationTools/main)
It takes a minute to load the environment.

### Locally

You can also run the tutorials and tools locally in a virtual environment, i.e., `venv` or
[miniconda](https://docs.conda.io/en/latest/miniconda.html).

1. Create and activate a new environment, e.g., on Mac/Linux terminal with `venv`:
   ```
   python -m venv pvdeg
   . pvdeg/bin/activate
   ```
   or with `conda`:
   ```
   conda create -n pvdeg
   conda activate pvdeg
   ```

1. Install `pvdeg` into the new environment with `pip`:
   ```
   python -m pip install pvdeg
   ```

1. Start a Jupyter session:

   ```
   jupyter notebook
   ```

1. Use the file explorer in Jupyter lab to browse to `tutorials`
   and start the first Tutorial.


Documentation
=============

Full API documentation is available at [ReadTheDocs](https://PVDegradationTools.readthedocs.io) where you can find detailed information on all functions, classes, and modules.
