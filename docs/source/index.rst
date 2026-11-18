.. pvdeg documentation master file, created by
   sphinx-quickstart on Thu Jan 18 15:25:51 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. .. image:: ../../tutorials_and_tools/pvdeg_logo.png
..    :width: 500

.. .. image:: ./_static/logo-vectors/PVdeg-Logo-Horiz-Color.svg


Welcome to PVDeg!
==============================================================

PVDeg is an open-source Python package for modeling photovoltaic (PV) degradation, developed at the National Laboratory of the Rockies (NLR) and supported by the Durable Module Materials (DuraMAT) consortium. It provides modular functions, materials databases, and calculation workflows for simulating degradation mechanisms (e.g., LeTID, hydrolysis, UV exposure) using weather data from the National Solar Radiation Database (NSRDB) and the Photovoltaic Geographical Information System (PVGIS). By integrating Monte Carlo uncertainty propagation and geospatial processing, PVDeg enables field-relevant predictions and uncertainty quantification of module reliability and lifetime.

The source code for PVDeg is hosted on `github <https://github.com/NatLabRockies/PVDegradationTools>`_. Please see the :ref:`installation` page for installation help.

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

If you use PVDeg in published work, please cite both the paper and the software.

**Recommended citation for PVDeg:**

   | Daxini, R., Ovaitt, S., Springer, M., Ford, T., & Kempe, M. (2026).
   | "PVDeg: a Python package for modeling degradation of solar photovoltaic systems."
   | *Journal of Open Source Software*, 11(127), 10617.
   | https://doi.org/10.21105/joss.10617

.. code-block:: bibtex

   @article{pvdeg_joss,
     author    = {Daxini, Rajiv and Ovaitt, Silvana and Springer, Martin and
                  Ford, Tobin and Kempe, Michael},
     title     = {{PVDeg: a Python package for modeling degradation of solar
                  photovoltaic systems}},
     journal   = {Journal of Open Source Software},
     publisher = {The Open Journal},
     year      = {2026},
     volume    = {11},
     number    = {127},
     pages     = {10617},
     doi       = {10.21105/joss.10617},
     url       = {https://doi.org/10.21105/joss.10617}
   }

**Citation for the latest PVDeg software version:**

   | Springer, M., Ovaitt, S., Daxini, R., Ford, T., Brown, M., Karas, J., Holsapple, D., & Kempe, M. (2026).
   | "PVDeg: a Python package for modeling degradation of solar photovoltaic systems" [Computer software].
   | *Zenodo*.
   | https://doi.org/10.5281/zenodo.8088382

.. code-block:: bibtex

   @software{pvdeg,
     author    = {Springer, Martin and Ovaitt, Silvana and Daxini, Rajiv and
                  Ford, Tobin and Brown, Matthew and Karas, Joseph and
                  Holsapple, Derek and Kempe, Michael},
     title     = {{PVDeg: a Python package for modeling degradation of solar
                  photovoltaic systems}},
     publisher = {Zenodo},
     year      = {2026},
     doi       = {10.5281/zenodo.8088382},
     url       = {https://doi.org/10.5281/zenodo.8088382}
   }

The Zenodo DOI above always resolves to the latest release. To cite the exact version you used, pick it from the version list on the `Zenodo page <https://doi.org/10.5281/zenodo.8088382>`_, or use the **"Cite this repository"** button on the `GitHub repository <https://github.com/NatLabRockies/PVDegradationTools>`_.

If you use PVDeg in a commercial or publicly-available application, please consider displaying the PVDeg logo:

.. image:: ./_static/logo-vectors/PVdeg-Logo-Color.svg
   :height: 150
   :alt: PVDeg logo (stacked)

.. image:: ./_static/logo-vectors/PVdeg-Logo-Horiz-Color.svg
   :width: 300
   :alt: PVDeg logo (horizontal)

Black, white, and color versions (SVG) are available in
`docs/source/_static/logo-vectors <https://github.com/NatLabRockies/PVDegradationTools/tree/main/docs/source/_static/logo-vectors>`_.


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
