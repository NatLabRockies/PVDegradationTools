# %%
import pvdeg

# %% [markdown]
# ## Define Geospatial Scenario Object
#
# To preform geospatial analysis we can create a `GeospatialScenario` object. Alternatively, to preform single location analysis use `Scenario`. Scenario and GeospatialScenario are generalized classes that can be used to replace the legacy functional pvdeg analysis approach with an object orented one.

# %%
geospatial_standoff_scenario = pvdeg.GeospatialScenario(
    name="standoff geospatial",
)

# %% [markdown]
# ## Add Location
# To add locations for geospatial analysis we will use the ``.addLocation`` method. We can choose downselect from the NSRDB to a country, state and county in that order. *Support for multiple of each category in list form soon.* The ``see_added`` flag allows us to see the gids we have added to the scenario.

# %%
geospatial_standoff_scenario.addLocation(
    state="Colorado", county="Summit", see_added=True, downsample_factor=3
)

# %% [markdown]
# ## Add Functions to the Pipeline
# The scenario has a queue of jobs to preform. These are stored in an attribute called ``pipeline``, you can directly update the pipeline but this will bypass the assistance given in creating the job function and parameters. The easiest way to add a job to the pipeline is the ``.updatePipeline`` method. For geospatial analysis, weather and metadata is collected and stored in the scenario at the time of the ``.addLocation`` method call so we do not need to include it below, but if we have other function kwargs to include, they should go in the ``func_params`` argument.
#
# Only a few pvdeg functions are currently supported for geospatial analysis. See the docstring for ``.updatePipeline`` to view currently supported functions. ``updatePipeline`` will not let you add unsupported geospatial functions. The ``see_added`` flag allows us to see the new job added to the pipeline.

# %%
geospatial_standoff_scenario.addJob(func=pvdeg.standards.standoff, see_added=True)

# %%
geospatial_standoff_scenario

# %% [markdown]
# ## Run the job in the pipeline
#
# Currently ``scenario`` only supports one geospatial analysis at a time. We cannot have two geospatial jobs at the same time.

# %%
geospatial_standoff_scenario.run()

# %% [markdown]
# ## Directly Access Results Attribute
#
# We can either view the results of the scenario pipeline using ``.viewScenario`` as shown above. The results will be displayed only if the pipeline has been run. Alternatively, we can directly view the ``results`` atribute of the scenario.

# %%
geospatial_standoff_scenario.results

# %% [markdown]
# ## Cleanup
#
# The scenario object will store its attributes in a file the python script's current working directory. If we want to delete this file when we are done with the scenario instance we can use the ``.clean()`` method as shown below.

# %%
geospatial_standoff_scenario.clean()

# %% [markdown]
# ## Example Geospatial Functionality
# Many functions are supported for geospatial analysis, here are a few.
# - ``pvdeg.standards.standoff``
# - ``pvdeg.humidity.module``
# - ``pvdeg.letid.calc_letid_outdoors``
#
# See the Geospatial Templates tutorial for an example on this.

# %%
geospatial_humidity_scenario = pvdeg.GeospatialScenario(
    name="humidity scenario", geospatial=True
)

geospatial_humidity_scenario.addLocation(
    state="Colorado", county="Jefferson", see_added=True
)

geospatial_humidity_scenario.addJob(
    func=pvdeg.humidity.module,
    func_params={
        "backsheet_thickness": 0.3,  # mm, thickness of PET backsheet
        "back_encap_thickness": 0.46,  # mm, thickness of EVA backside encapsulant
        "encapsulant": "W001",  # EVA encapsulant
        "backsheet": "W017",  # PET backsheet
    },
    see_added=True,
)

# %%
geospatial_humidity_scenario.run()

# %%
geospatial_humidity_scenario.results
