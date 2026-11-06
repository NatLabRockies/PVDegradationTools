API
===

PVDeg is organized as a layered architecture built around a common data convention. At
its foundation is a library of modular *core functions*, each modeling a single
environmental stressor (e.g., temperature, irradiance, relative humidity) or degradation
mechanism for one location from a standardized weather DataFrame and location metadata.
The core functions share this input convention and return either a timeseries or a
summary numeric, therefore a user may call one in isolation or chain the output of one
(e.g., module temperature) into another (e.g., a moisture-ingress or LeTID model) to
assemble a bottom-up lifetime prediction. Three higher-level layers then orchestrate
these same functions without re-implementing their physics: the `Scenario` class
sequences them into reproducible, multi-step pipelines; the geospatial layer vectorizes
them across latitude–longitude grids; and the Monte Carlo engine propagates parameter
uncertainty through them. This reuse is enabled by a lightweight internal decorator
(`@geospatial_quick_shape`) that records each function's output structure, so code
written for a single location can be automatically templated and scaled to continental
or global analyses without modification. The subsections below describe these layers in
turn, together with the tutorials and open datasets that support them.

.. currentmodule:: pvdeg

.. autosummary::
    :toctree: _autosummary/
    :template: module.rst

    collection
    degradation
    design
    fatigue
    humidity
    letid
    montecarlo
    pysam
    scenario
    geospatial
    spectral
    standards
    symbolic
    temperature
    utilities
    weather
