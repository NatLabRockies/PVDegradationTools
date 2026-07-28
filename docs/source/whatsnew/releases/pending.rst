v0.8.0 (unreleased)
===================

Enhancements
------------
- Efficient geospatial NSRDB loading. ``pvdeg.weather.get`` and
  ``pvdeg.weather.get_NSRDB`` (with ``geospatial=True``) now accept ``bbox``,
  ``downsample``, and ``land_only`` arguments and read only the requested sites
  instead of the full grid, making it practical to load very large NSRDB grids
  such as GOES ``full_disc`` (~9.5 million sites). The target site GIDs are
  resolved from a fast coordinate-only read via the new
  ``pvdeg.weather.nsrdb_gids``. (:pull:`351`)
- Added the ``"Polar"`` NSRDB satellite (the Arctic ``polar`` grid) to the
  satellite map, and a ``land_only`` option that drops offshore sites using the
  NSRDB ``offshore`` flag. (:pull:`351`)
- Added timeseries downsampling to the NSRDB geospatial loader through a
  ``resample`` argument, which block-averages the time axis with xarray
  ``coarsen`` to keep the dask graph small. (:pull:`351`)
- ``pvdeg.geospatial.analysis`` gained a ``gid_chunk`` argument and now
  auto-chunks the weather dataset along ``gid`` when it has a single chunk, so
  the analysis runs one task per location in parallel instead of serially on a
  single worker. (:pull:`351`)


Documentation
-------------
- Updated the geospatial tutorials (``tutorials/05_geospatial``) so the
  templates and outdoor LETID demonstrations run against the current API.
  (:pull:`351`)
- Reorganized the tutorials into numbered categories (``01_basics`` through
  ``06_advanced``, plus ``10_workshop_demos`` and ``tools``). Removed leftover
  duplicate notebooks and orphaned scripts left by the geospatial/scenario
  split, fixed the internal cross-reference links between tutorials, and
  updated the tutorial listing in the README and docs. (:pull:`351`)
- Added a ``tutorials/06_advanced/02_pysam_single_location`` tutorial
  demonstrating a single-location PySAM ``pvsamv1`` simulation from PVGIS
  weather. (:pull:`351`)


Deprecations
-------------


Bug Fixes
---------
- Fixed a numerical blow-up in the LETID model at very low temperatures.
  ``pvdeg.letid.calc_letid_outdoors`` and ``pvdeg.letid.calc_letid_lab``
  integrate the three-state defect kinetics with an explicit forward-Euler step
  that could overshoot under extreme cold (e.g. Arctic / Siberian winter, module
  temperatures below about -60 °C), driving the defect-state populations far
  outside their physical range and collapsing the carrier lifetime (normalized
  power crashing to ~0.56). The state populations ``NA``, ``NB``, and ``NC`` are
  now clamped to ``[0, 100]`` at each step. (:pull:`351`)
- Fixed ``pvdeg.geospatial.analysis`` dropping the ``time`` dimension for
  functions that return a timeseries ``pandas.Series`` (for example
  ``pvdeg.temperature.cell``), which raised
  ``ValueError: Dimensions {'time'} missing on returned object``. The weather
  index name is now preserved when its dtype is coerced, and a ``Series`` result
  is named after the function's declared geospatial shape so that it matches the
  output template. (:pull:`351`)
- Fixed the Lambert projection in the geospatial map plots so that the data and
  the map features share a single set of axes and register correctly.
  (:pull:`351`)
- Continued the NREL to NLR migration: renamed the internal
  ``pvdeg.utilities.nrel_kestrel_check`` to ``pvdeg.utilities.nlr_kestrel_check``
  and updated the Kestrel hostname to the ``nlr.gov`` domain. (:pull:`351`)
- Fixed ``pvdeg.geospatial.analysis`` raising
  ``AttributeError: 'Future' object has no attribute 'loc'`` whenever a Dask
  client was active. ``analysis`` scatters ``meta_df`` to a broadcast
  ``distributed.Future`` to keep the task graph small, but xarray's
  ``map_blocks`` does not auto-resolve Futures passed through ``kwargs``;
  ``calc_block`` now materializes the Future before use. (:pull:`351`)
- Fixed ``pvdeg.utilities._add_material`` re-serializing the material database
  JSON files with escaped non-ASCII characters and no trailing newline, which
  produced spurious diffs in the packaged data files (for example
  ``pvdeg/data/O2permeation.json``). It now writes UTF-8 with a trailing
  newline. (:pull:`351`)


Dependencies
------------


Contributors
------------
- Martin Springer (:ghuser:`martin-springer`)
