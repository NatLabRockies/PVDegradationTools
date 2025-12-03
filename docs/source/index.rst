.. pvdeg documentation master file, created by
   sphinx-quickstart on Thu Jan 18 15:25:51 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. .. image:: ../../tutorials_and_tools/pvdeg_logo.png
..    :width: 500

.. image:: ./_static/logo-vectors/PVdeg-Logo-Horiz-Color.svg


Welcome to PVDeg!
==============================================================

PVDeg is an open-source Python package for modeling photovoltaic (PV) degradation, developed at the National Renewable Energy Laboratory (NREL) and supported by the Durable Module Materials (DuraMAT) consortium. It provides modular functions, materials databases, and calculation workflows for simulating degradation mechanisms (e.g., LeTID, hydrolysis, UV exposure) using weather data from the National Solar Radiation Database (NSRDB) and the Photovoltaic Geographical Information System (PVGIS). By integrating Monte Carlo uncertainty propagation and geospatial processing, PVDeg enables field-relevant predictions and uncertainty quantification of module reliability and lifetime.

The source code for PVDeg is hosted on `github <https://github.com/NREL/pvdeg>`_. Please see the :ref:`installation` page for installation help.

See :ref:`tutorials` to learn how to use and experiment with various functionalities


.. image::  ./_static/PVDeg-Flow.svg
    :alt: PVDeg-Flow diagram.


Key Features
============

- **Core Degradation Functions**: Dedicated functions for physical degradation mechanisms including moisture ingress, LeTID, UV exposure, and thermal stress
- **Scenario Class**: Simplified workflow interface for complex multi-parameter degradation studies
- **Geospatial Analysis**: Large-scale spatial analyses with parallel processing across geographic regions
- **Monte Carlo Framework**: Uncertainty quantification through parameter distribution sampling
- **Material Databases**: Curated degradation parameters, kinetic coefficients, and material properties
- **Weather Data Integration**: Seamless access to NSRDB and PVGIS meteorological data
- **Standards Support**: Contributions to IEC TS 63126 and other standardization efforts

How the Model Works
===================

PVDeg's core API provides dedicated functions for calculating physical degradation mechanisms, accessing material properties and environmental stressors. These functions rely on standardized environmental stressors such as temperature, irradiance, and humidity, and can be chained to produce lifetime predictions under realistic field conditions.

To simplify complex workflows, PVDeg wraps its core functions into a ``Scenario`` class that defines locations, module configurations, and degradation mechanisms. This enables user-friendly workflows, simplifying the setup and execution of complex multi-parameter degradation studies.

The geospatial analysis layer enables large-scale spatial analyses by automatically distributing degradation calculations across geographic regions using parallel processing and advanced data structures. It integrates environmental data from NSRDB and PVGIS and automates sampling across latitude-longitude grids to produce maps, such as standoff distance distribution used in IEC TS 63126 compliance studies.

PVDeg's Monte Carlo engine samples parameter distributions and their correlations to generate thousands of realizations, producing confidence intervals on degradation rates rather than single deterministic values. This capability can help quantify uncertainty in complex and non-linear module lifetime predictions, and identify which parameters most strongly affect reliability risk.

Citing PVDeg
============

If you use PVDeg in a published work, please cite both the software and the paper.

**Software Citation:**

Click the "Cite this repository" button on the `GitHub repository <https://github.com/NREL/PVDegradationTools>`_, or visit `Zenodo <https://zenodo.org/records/8088578/latest>`_ for the DOI corresponding to your specific version. On the Zenodo page, use the "Cite as" section in the right sidebar to copy the citation in your preferred format (BibTeX, APA, etc.).

**JOSS Paper (In Review):**

.. code-block::

   Daxini, R., Ovaitt, S., Springer, M., Ford, T., & Kempe, M. (2025). PVDeg: a python package for modeling degradation on solar photovoltaic systems. Journal of Open Source Software (In Review).

**Software Record:**

.. code-block::

   Holsapple, Derek, Ayala Pelaez, Silvana, Kempe, Michael. "PV Degradation Tools", NREL Github 2020, Software Record SWR-20-71.


.. toctree::
   :hidden:
   :titlesonly:

   user_guide/index
   tutorials/index
   api
   whatsnew/index

..
   Indices and tables
   ==================

   * :ref:`genindex`
   * :ref:`modindex`
   * :ref:`search`
