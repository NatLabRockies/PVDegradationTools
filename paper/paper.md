---
title: 'PVDeg: a python package for modeling degradation on solar photovoltaic systems'
tags:
  - Python
  - solar energy
  - photovoltaics
  - renewable energy
  - degradation
authors:
  - name: Rajiv Daxini
    orcid: 0000-0003-1993-9408
    affiliation: 1
  - name: Silvana Ovaitt
    orcid: 0000-0003-0180-728X
    affiliation: 1
  - name: Martin Springer
    orcid: 0000-0001-6803-108X
    affiliation: 1
  - name: Tobin Ford
    orcid: 0009-0000-7428-5625
    affiliation: 1
  - name: Michael Kempe
    orcid: 0000-0003-3312-0482
    affiliation: 1
affiliations:
 - name: National Renewable Energy Laboratory (NREL)
   index: 1
date: 01 November 2025
bibliography: paper.bib
---

# Summary

PVDeg is an open-source Python package for modeling photovoltaic (PV) degradation, developed at the National Renewable Energy Laboratory (NREL) and supported by the Durable Module Materials (DuraMAT) consortium.  It provides modular functions and workflows for simulating degradation mechanisms (e.g., LeTID, hydrolysis, UV exposure) using weather data from the National Solar Radiation Database (NSRDB) and the Photovoltaic Geographical Information System (PVGIS).  By integrating Monte Carlo uncertainty propagation and geospatial processing, PVDeg enables field-relevant predictions and uncertainty quantification of module reliability and lifetime.

PVDeg is developed openly on GitHub and releases are distributed via the Python Package Index (PyPi). The source code is freely available under under the BSD 3-Clause license, and copyrighted by the Alliance for Sustainable Energy allowing permissive use with attribution.  PVDeg follows best practices for open-source python software, with a robust testing framework across Python 3.x environments, semantic versioning, and a full supproting documentation available at pvdegradationtools.readthedocs.io.

# Statement of Need

As PV deployment expands, especially into new and demanding operational environments, material degrdation poses a challenge to the lifetime of PV modules.  Modeling degradation is crucial for anticipating performance losses, guiding material selection, and enabling proactive maintenance strategies that extend the operational lifetime of PV modules in diverse environments.  Existing PV modeling tools such as pvlib-python [@pvlib] and SAM [@SAM] are able to simulate system energy yield, but not degradation.  PVDeg fills this gap by providing modular degradation models, material databases, and uncertainty quantification workflows.  PVDeg supports both research and industry use by automating degradation modeling, enabling reproducible studies of module lifetime nad performance worldwide.  It also supports ongoing standardization work, including contributions to IEC TS 63126 [@IEC63126].  PVDeg is an important component of a growing ecosystem of open-source tools for solar energy [@Holmgren2018].

![Example of geospatial degradation modeling in PVDeg: (a) calculated standoff distances for IEC TS 63126 across the continental U.S.\label{fig:visualization}](IECTS_63126.PNG){ width=80% }

# Software Functionality

## Core Functions
The core API provides dedicated functions for calcualting physical degraation mechanisms, accessing material properties and environmental stressors.  Examples include `pvdeg.humidity.module()` for moisture ingress modeling [@picket2013hydrolysis], and `pvdeg.letid.calc_letid_outdoors()` for modeling light and elevated temperature induced degradation (LeTID) [@karas2022letidstudy; @repins2023longterm].   These functions rely on standardized environmental drivers such as temperature, irradiance, and humidity, and can be chained to produce lifetime predictions under realistic field conditions.

## Scenario Class
To simplify complex workflows, PVeg wraps its core functions into a ``Scenario`` class that defines locations, module configurations, and degradation mechanisms.  This enables user-friendly workflows, simplifying the setup and execution of complex multi-parameter degradation studies.  This layer provides an intuitive interface for multiple analysis of different degrdation,climates, nad conigurations for comarative analysis.  Tutorials in Jupyter notebooks and hosted examples on *Read the Docs* demonstrate full end-to-end analyses.

## Geospatial Analysis
The geospatial analysis layer enables large-scale spatial analyses by automatically distributing degradation calculations across geographic regions using parallel processing and advanced data structures.  It integrates environmental data from NSRDB and PVGIS and automates sampling across latitude-longitude grids to produce degradation maps, such as standoff distance distribution used in IEC TS 63126 compliance studies [@IEC63126].  The geospatial layer includes specialized visualization functions for mapping results and supports both uniform and stochastic spatial sampling strategies to balance computational efficiency with geographic coverage.  Parallelization routines are compatible with NREL's open-source *GeoGridFusion* framework [@ford2025geogridfusion; @Tobin2025geogridfusion], allowing users to down-select meteorological datasets efficiently adn execute computations without high-performance computing access.  This capability supports national— and global-scale analyses of degradation phenomena.

## Monte Carlo Framework

Laboratory-to-field extrapolation carries significant uncertainty in kinetic parameters.  PVDeg’s Monte Carlo engine samples parameter distributions and their correlations to generate thousands of realizations, producing confidence intervals on degradation rates rather than single deterministic values.  This capability, described in [@springer2022futureproofing], helps quantify uncertainty in module lifetime predictions and identify which parameters most strongly affect reliability risk.

## Tutorials and Tools
The tutorials and tools component of PVDeg consists of a comprehensive suite of Jupyter notebooks that demonstrate practical workflows for modeling PV degradation.  These notebooks cover core degradation mechanisms, scenario setup, geospatial analysis, and uncertainty quantification, providing step-by-step guidance for both new and advanced users.  Each tutorial is designed to be interactive and reproducible, enabling users to explore real-world datasets, customize parameters, and visualize results.  The notebooks supprot comparative studies and integration with external meteorological data sources such as NSRDB and PVGIS.  By leveraging these notebooks, users can efficiently learn, apply, and extend PVDeg’s capabilities for research and industry applications.

## Open datasets
A growing component of PVDeg is its compilation of community-driven open datasets for PV degradation modeling.  These databases include curated degradation parameters and material property data, such as kinetic coefficients for common degradation mechanisms and permeation properties for materials (e.g., H₂O, O₂, acetic acid).  The datasets are continuously expanded and updated, serving as a growing resource for users to access validated values for modeling and analysis.  Users are encouraged to contribute their own data, enhancing the collective knowledge base and supporting reproducible research.  The core PVDeg API also provides users a means to seamlessly query these datasets and use
them in their own modeling workflows, analysis, and investigations.  The development and maintenance of these degradation databases and associated API calls also supprots reproducible, relaiable, and field-relevaent degradation modeling for the PV community.

# Example Applications

Since its first release as PV Degradation Tools [@Holsapple2020pvdegtools], PVDeg has been adopted in multiple studies across the PV reliability community:
* Thermal Stability and IEC TS 63126 Compliance: Used to calculate effective standoff distances and generate public maps supporting the IEC TS 63126 standard [@IEC63126].
* Light and Elevated Temperature Induced Degradation (LeTID): Integrated into the international interlaboratory comparison study of LeTID effects in crystalline-silicon modules [@karas2022letidstudy] and follow-up analyses of field-aged arrays [@repins2023longterm; @karas2024letid].
* Geospatial Performance Modeling: Coupled with GeoGridFusion [@ford2025geogridfusion] to streamline weather-data storage and spatial queries for large-scale degradation simulations.
* Agrivoltaic and System-Level Modeling: Combined with PySAM [@SAM] to assess degradation-driven yield losses and ground-irradiance patterns in dual-use agrivoltaic systems.  [@OvaittPuertoRico2023]
* Material-Property Parameterization: Leveraged in studies of UV-induced polymer degradation [@kempe2023uvstress] and moisture-related failures in encapsulants and backsheets [@coyle2011cigs].

These applications highlight PVDeg’s versatility as the “PV Library of degradation” — an open, community-driven platform linking materials science, environmental modeling, and field performance.

# Ongoing Development

Version 0.6.2 XX CITATION XX is the latest stable release, incorporating XXX summary line or two XXX. Future work includes expanding the degradation and material parameter databases using large language model driven literature searches, and enhancing the Scenario class to enable handling multiple materials and degradation pathways within the same workflow. This will mitigate the need for users to design and execute Scenarios for different degradation degradation pathways and materials.


# Acknowledgements

We acknowledge all code, documentation, and discussion contributors to the PVDeg project.

This work was authored by the National Renewable Energy Laboratory, operated by Alliance for Sustainable Energy, LLC, for the U.S.  Department of Energy (DOE) under Contract No.  DE-AC36-08GO28308.  Funding provided as part of the Durable Modules Materials Consortium (DuraMAT), an Energy Materials Network Consortium funded by the U S Department of Energy, Office of Energy Efficiency and Renewable Energy, Solar Energy Technologies Office Agreement Number 32509.  The research was performed using computational resources sponsored by the Department of Energy's Office of Energy Efficiency and Renewable Energy and located at the National Renewable Energy Laboratory.  The views expressed in the article do not necessarily represent the views of the DOE or the U.S.  Government.  The U.S.  Government retains and the publisher, by accepting the article for publication, acknowledges that the U.S.  Government retains a nonexclusive, paid-up, irrevocable, worldwide license to publish or reproduce the published form of this work, or allow others to do so, for U.S.  Government purposes.


# References
