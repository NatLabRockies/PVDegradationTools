<img src="https://raw.githubusercontent.com/NREL/PVDegradationTools/refs/heads/main/docs/source/_static/logo-vectors/PVdeg-Logo-Horiz-Color.svg" width="600">


<table>
<tr>
  <td>License</td>
  <td>
    <a href="https://github.com/NREL/PVDegradationTools/blob/master/LICENSE.md">
    <img src="https://img.shields.io/pypi/l/pvlib.svg" alt="license" />
    </a>
</td>
</tr>
<tr>
  <td>Publications</td>
  <td>
     <a href="https://zenodo.org/records/8088578/latest"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.8088578.svg" alt="DOI"></a>
  </td>
</tr>
<tr>
  <td>Documentation</td>
  <td>
	<a href='https://PVDegradationTools.readthedocs.io'>
	    <img src='https://readthedocs.org/projects/pvdegradationtools/badge/?version=stable' alt='Documentation Status' />
	</a>
  </td>
</tr>
<tr>
  <td>Build status</td>
  <td>
   <a href="https://github.com/NREL/PVDegradationTools/actions/workflows/pytest.yml?query=branch%3Amain">
      <img src="https://github.com/NREL/PVDegradationTools/actions/workflows/pytest.yml/badge.svg?branch=main" alt="GitHub Actions Testing Status" />
   </a>
   <a href="https://codecov.io/gh/NREL/PVDegradationTools" >
   <img src="https://codecov.io/gh/NREL/PVDegradationTools/graph/badge.svg?token=4I24S8BTG7"/>
   </a>
  </td>
</tr>
</table>



# PVDeg: Python Package for Modeling Degradation in Photovoltaic Systems

PVDeg is an open-source Python package for modeling photovoltaic (PV) degradation, developed at the National Renewable Energy Laboratory (NREL) and supported by the Durable Module Materials (DuraMAT) consortium. It provides modular functions, materials databases, and calculation workflows for simulating degradation mechanisms (e.g., LeTID, hydrolysis, UV exposure) using weather data from the National Solar Radiation Database (NSRDB) and the Photovoltaic Geographical Information System (PVGIS). By integrating Monte Carlo uncertainty propagation and geospatial processing, PVDeg enables field-relevant predictions and uncertainty quantification of module reliability and lifetime.

## Key Features

- **Core Degradation Functions**: Dedicated functions for physical degradation mechanisms including moisture ingress, LeTID, UV exposure, and thermal stress
- **Scenario Class**: Simplified workflow interface for complex multi-parameter degradation studies
- **Geospatial Analysis**: Large-scale spatial analyses with parallel processing across geographic regions
- **Monte Carlo Framework**: Uncertainty quantification through parameter distribution sampling
- **Material Databases**: Curated degradation parameters, kinetic coefficients, and material properties
- **Weather Data Integration**: Seamless access to NSRDB and PVGIS meteorological data
- **Standards Support**: Contributions to IEC TS 63126 and other standardization efforts
## Example Applications

PVDeg has been adopted in multiple studies across the PV reliability community:

- **Thermal Stability and IEC TS 63126 Compliance**: Calculate effective standoff distances and generate public maps supporting the IEC TS 63126 standard
- **Light and Elevated Temperature Induced Degradation (LeTID)**: Integrated into international interlaboratory comparison studies and field-aged array analyses
- **Geospatial Performance Modeling**: Coupled with GeoGridFusion to streamline weather-data storage and spatial queries for large-scale degradation simulations
- **Agrivoltaics and System-Level Modeling**: Combined with PySAM to assess degradation-driven yield losses in dual-use agrivoltaic systems
- **Material-Property Parameterization**: Studies of UV-induced polymer degradation and moisture-related failures in encapsulants and backsheets

Tutorials
=========

### Jupyter Book

For in depth tutorials you can run online, see our [Jupyter Book](https://nrel.github.io/PVDegradationTools/intro.html)

**Development Preview:** Preview the latest development changes at [dev-preview](https://nrel.github.io/PVDegradationTools/dev-preview/intro.html)

Clicking on the rocket-icon on the top allows you to launch the notebooks on [Google Colaboratory](https://colab.research.google.com/) for interactive mode.
Just uncomment the first line `pip install ...` to install the environment on each notebook if you follow this mode.

### Binder

To run these tutorials in Binder, you can click here:
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/NREL/PVDegradationTools/main)
It takes a minute to load the environment.

### Locally

You can also run the tutorial locally in a virtual environment, i.e., `venv` or
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
   and start the first Tutorial. Tutorials are organized into the following categories:
   - `01_basics/` - Introduction to PVDeg fundamentals
   - `02_degradation/` - Degradation mechanism modeling (LID, LeTID, Van't Hoff)
   - `03_monte_carlo/` - Monte Carlo uncertainty analysis
   - `04_geospatial/` - Geospatial and HPC scenarios (includes NREL HPC workflows)
   - `05_advanced/` - Advanced topics and API access
   - `10_workshop_demos/` - Workshop and demonstration notebooks
   - `tools/` - Standalone analysis and calculation tools


Documentation
=============

Full API documentation is available at [ReadTheDocs](https://PVDegradationTools.readthedocs.io) where you can find detailed information on all functions, classes, and modules.


Installation
============

PVDeg releases may be installed using the `pip` and `conda` tools. Compatible with Python 3.10 and above.

Install with:

    pip install pvdeg

### Optional Dependencies

PVDeg offers optional dependency groups for specific use cases:

    pip install pvdeg[sam]      # Install with PySAM support
    pip install pvdeg[docs]     # Install documentation tools
    pip install pvdeg[test]     # Install testing tools
    pip install pvdeg[books]    # Install Jupyter Book tools
    pip install pvdeg[all]      # Install all optional dependencies

### Developer Installation

For developer installation, clone the repository, navigate to the folder location and install as:

    pip install -e .[all]

Running jupyter notebooks using anaconda prompt
===============================================

Note that in order to run notebooks cleanly and validate outputs, use the following
commands to run either one notebook:

    jupyter nbconvert --to notebook --execute --inplace "tutorials/01_basics/01_basics_humidity_design.ipynb"

or all notebooks inside a specific tutorial category:

    jupyter nbconvert --to notebook --execute --inplace "tutorials/01_basics/*.ipynb"

This avoids formatting issues that may arise depending on your own local environment
or IDE.


License
=======

[BSD 3-clause](https://github.com/NREL/PVDegradationTools/blob/main/LICENSE.md)


Contributing
============

We welcome contributions to this software. Please read the copyright license agreement (cla-1.0.md), with instructions on signing it in sign-CLA.md.

All code, documentation, and discussion contributors are acknowledged for their contributions to the PVDeg project.


Getting support
===============

If you suspect that you may have discovered a bug or if you'd like to
change something about PVDeg, then please make an issue on our
[GitHub issues page](https://github.com/NREL/PVDegradationTools/issues).


Citing
======

If you use PVDeg in a published work, please cite:

**JOSS Paper (In Review):**

	Daxini, R., Ovaitt, S., Springer, M., Ford, T., & Kempe, M. (2025). PVDeg: a python package for modeling degradation on solar photovoltaic systems. Journal of Open Source Software (In Review).

**Latest Release:**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8088578.svg)](https://zenodo.org/records/8088578/latest)

**Software Record:**

	Holsapple, Derek, Ayala Pelaez, Silvana, Kempe, Michael. "PV Degradation Tools", NREL Github 2020, Software Record SWR-20-71.


Acknowledgements
================

This work was authored by the National Renewable Energy Laboratory, operated by Alliance for Sustainable Energy, LLC, for the U.S. Department of Energy (DOE) under Contract No. DE-AC36-08GO28308. Funding provided as part of the Durable Modules Materials Consortium (DuraMAT), an Energy Materials Network Consortium funded by the U.S. Department of Energy, Office of Energy Efficiency and Renewable Energy, Solar Energy Technologies Office Agreement Number 32509. The research was performed using computational resources sponsored by the Department of Energy's Office of Energy Efficiency and Renewable Energy and located at the National Renewable Energy Laboratory. The views expressed in the article do not necessarily represent the views of the DOE or the U.S. Government. The U.S. Government retains and the publisher, by accepting the article for publication, acknowledges that the U.S. Government retains a nonexclusive, paid-up, irrevocable, worldwide license to publish or reproduce the published form of this work, or allow others to do so, for U.S. Government purposes.
