<img src="PVD_logo.png" width="100">

# PV Degradation Tools - Tutorials (pvdeg)

## Tutorial Organization

Tutorials are organized by topic into the following folders:

### 📚 [Basics](basics/)
Introduction to PVDegradationTools - start here if you're new!
- Basic concepts, humidity, and design
- Degradation modeling overview
- Spectral degradation
- Weather database access

### 🔬 [Degradation](degradation/)
Specialized degradation mechanisms (LID, LETID, Van't Hoff)
- B-O LID and LETID accelerated testing
- Outdoor degradation prediction
- Degradation kinetics

### 🎲 [Monte Carlo](monte_carlo/)
Uncertainty analysis and probabilistic modeling
- Arrhenius model simulations
- Standoff parameter analysis

### 🗺️ [Geospatial](geospatial/)
Large-scale spatial analysis and scenarios
- ⚠️ Most tutorials require **HPC access**
- Local scenario tutorial available without HPC

### 🚀 [Advanced](advanced/)
Custom functions and API integrations
- ⚠️ Some tutorials require **API keys** (NSRDB, PVGIS)
- Custom function development
- PySAM integration

### 🎪 [Workshop Demos](workshop_demo/)
Live demonstration materials from workshops

---

## Running the Tutorials

### Jupyter Book (Recommended)

For learning to use PVDeg through our tutorials online, see our [jupyter-book](https://nrel.github.io/PVDegradationTools/intro.html)
Clicking on the rocket-icon on the top allows you to launch the notebooks on [Google Colaboratory](https://colab.research.google.com/) for interactive mode.
Just uncomment the first line `pip install ...` to install the environment on each notebook if you follow this mode.

### Binder

To run these tutorials in Binder, you can click here:
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/NREL/PVDegradationTools/main)
It takes a minute to load the environment.

### Locally

You can also run the tutorials locally in a virtual environment, i.e., `venv` or
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

2. Install `pvdeg` into the new environment with `pip`:
   ```
   python -m pip install pvdeg
   ```

3. Start a Jupyter session:
   ```
   jupyter notebook
   ```

4. Use the file explorer in Jupyter to browse to the tutorial category folder and start with the first tutorial.

---

## Special Requirements

### 🔑 API Keys Required
Some tutorials in the **Advanced** folder require free API keys:
- **NSRDB API**: Register at https://developer.nrel.gov/signup/

### 💻 HPC Access Required
Most tutorials in the **Geospatial** folder require High Performance Computing resources for:
- Large-scale geospatial computations
- Multi-location scenario analyses
- Distributed computing workflows

---

## Documentation

We also have documentation in [ReadTheDocs](https://PVDegradationTools.readthedocs.io) where you can find more details on the API functions.
