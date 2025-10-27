---
title: 'PVDeg: a python package for modeling degradation on solar photovoltaic systems'
tags:
  - Python
  - solar energy
  - photovoltaics
  - renewable energy
  - degradation mechanisms
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

[comment]: Brief Intro
PVDeg is an open-source Python package for modeling photovoltaic (PV) degradation, developed at NREL and supported by DuraMAT. It provides functions and workflows for simulating degradation mechanisms (e.g., LETID, hydrolysis, UV exposure) using weather data from NSRDB and PVGIS, enabling field-relevant predictions and uncertainty quantification through Monte Carlo methods.

# Statement of Need

Reliable lifetime prediction requires translating laboratory stress data to field conditions. Existing tools such as PVLib and SAM model system performance but not degradation. PVDeg fills this gap by providing modular degradation models, material databases, and uncertainty quantification workflows. PVDeg supports both research and industry use by automating degradation modeling, enabling reproducible studies of module lifetime nad performance worldwide. PVDeg is an important component of a growing ecosystem of open-source tools for solar energy [@Holmgren2018].

![Example of geospatial degradation modeling in PVDeg: (a) calculated standoff distances for IEC TS 63126 across the continental U.S.\label{fig:visualization}](IECTS_63126.PNG){ width=80% }

[comment]: Hosting and documentation
PVDeg is hosted on Github and PyPi, and it was developed by contributors from national laboratories, academia, and private industry. PVDeg is copyrighted by the Alliance for Sustainable Energy with a BSD 3-clause license allowing permissive use with attribution. PVDeg is extensively tested for functional and algorithm consistency. Continuous integration services check each pull request on Linux and Python versions 2.7 and 3.6. PVDeg is thoroughly documented, and detailed tutorials are provided for many features. The documentation includes help for installation and guidelines for contributions. The documentation is hosted at readthedocs.org. Github’s issue trackers provide venues for user discussions and help.

[comment]: Introducing the 3 parts: classes/functions, the library, and the geospatial
# Core Functions
The PVDeg python library is a spatio-temporal modeling assessment tool that empowers users to calculate various PV degradation modes, for different PV technologies and materials. It is designed to serve the PV community, including researchers, device manufacturers, and other PV  stakeholders to assess different degradation modes in locations around the world. The library is developed and hosted open-source on GitHub, and is structured in three layers: core functions and classes, scenario analysis class, and geospatial analysis. These algorithms are typically implementations of models published in the existing peer-reviewed literature. In addition, data for PVDeg is sourced from the National Solar Radiation Database and the (NSRDB) and Photovoltaic Geographical Information System (PVGIS). PVDeg also contains its own internal database of material and degradation parameters. The core API consists of functions and classes that provide specialized calculations for individual degradation mechanisms, material properties, and environmental modeling. Examples include `pvdeg.humidity.module()` for moisture ingress modeling, and `pvdeg.letid.calc_letid_outdoors()` for light and elevated temperature induced degradation. 

# Scenario Class
The scenario analysis class layer wraps the core API functions into user-friendly workflows, simplifying the setup and execution of complex multi-parameter degradation studies. This layer provides an intuitive interface for multiple analysis components, with practical implementation examples in a series of Jupyter notebook tutorials. Users can create `Scenario`  objects to define locations, modules, and analysis pipelines, then execute multiple degradation calculations simultaneously across different module configurations and extract results for comparative analysis. 

# Key Features: Geospatial and Monte Carlo Modeling
The geospatial analysis layer enables large-scale spatial analyses by automatically distributing degradation calculations across geographic regions using parallel processing and advanced data structures. This layer supports studies such as mapping standoff distances across the United States, analyzing LETID degradation patterns across climate zones, and identifying optimal locations for specific PV technologies. The geospatial layer includes specialized visualization functions for mapping results and supports both uniform and stochastic spatial sampling strategies to balance computational efficiency with geographic coverage.


[comment]: Release Info
Holsapple, Derek, Ayala Pelaez, Silvana, Kempe, Michael. "PV Degradation Tools", NREL Github 2020, Software Record SWR-20-71.

PVDeg was originally released in 2020 as "PV Degradation Tools" [@Holsapple], with major expansions for geospatial and Monte Carlo uncertainty modeling added in 2024 [REF]. Udated Zenodos... Additional features continue to be added as described in the documentation’s “What’s New” section.

# Example Applications

PVDeg has been used in numerous studies, for example, in [@ ], for calculating effective standoff distances across the US following IEC 63126 for thermal stability, as well as on the standard itself, IEC TS 63126, for providing user-friendly maps for calculating these standoff distances for any worldwide location. [@IEC TS 6t3126]. It has also been used for studying the senstivity of diffuse-tracing methodologies geospatially to time reponse and weather, modeling the yearly performance and tracker movement change under smart tracking algorithms  in [@Adinolfi2024] and [@Adinolfi2025]. It's most recent use is also on the implementation of geospatial modeling of pySAM simulatins for calculation of ground-irradiance for dual-use agrivoltaics [@PVSC 2023], wiht pending dataset publication. an example of geospatial and geo-temporal results can also be seen in [@Karas] study of the LeTID process phenomenon, where users can observe the output of module degradaiton and improvement over a period of various years to these recently discovered and parameterized degradation.


[comment]: Plans
Plans for bifacial_radiance development include the implementation of new and existing models, addition of functionality to assist with input/output, and improvements to API consistency.

# Acknowledgements

The authors acknowledge and thank the code, documentation, and discussion contributors to the project.

This work was authored by the National Renewable Energy Laboratory, for the U.S. Department of Energy (DOE) under Contract No. DE-AC36-08GO28308. Funding was provided by the U.S. Department of Energy’s Office of Energy Efficiency and Renewable Energy (EERE) under Solar Energy Technologies Office Agreement Number 34910.

# References