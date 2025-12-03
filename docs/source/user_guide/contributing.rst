.. _contributing:

Contributing
============

We welcome contributions to PVDeg! Whether you're fixing bugs, adding features,
improving documentation, or contributing to our material property databases, your
help is valuable to the PV community.

For a quick overview, see `CONTRIBUTING.md <https://github.com/NREL/PVDegradationTools/blob/main/CONTRIBUTING.md>`_ on GitHub.

This guide provides comprehensive details for contributors.


Easy Ways to Contribute
~~~~~~~~~~~~~~~~~~~~~~~

Here are ways to contribute, even if you're new to PVDeg, git, or Python:

* **Report bugs or request features** via `GitHub issues <https://github.com/NREL/PVDegradationTools/issues>`_
* **Join discussions** on existing issues and pull requests
* **Improve documentation** - fix typos, clarify explanations, add examples
* **Enhance unit tests** - increase coverage or improve test quality
* **Create or improve tutorials** - demonstrate PVDeg in your area of expertise
* **Contribute to material databases** - add validated degradation parameters and properties
* **Share your work** - add your project to our `wiki <https://github.com/NREL/PVDegradationTools/wiki>`_
* **Spread the word** - tell colleagues about PVDeg

Getting Started
~~~~~~~~~~~~~~~

Development Environment Setup
------------------------------

1. **Fork and clone the repository**:

   .. code-block:: bash

       git clone https://github.com/YOUR-USERNAME/PVDegradationTools.git
       cd PVDegradationTools

2. **Create a virtual environment**:

   Using venv:

   .. code-block:: bash

       python -m venv pvdeg-dev
       source pvdeg-dev/bin/activate  # Windows: pvdeg-dev\\Scripts\\activate

   Or using conda:

   .. code-block:: bash

       conda create -n pvdeg-dev python=3.11
       conda activate pvdeg-dev

3. **Install in editable mode with all dependencies**:

   .. code-block:: bash

       pip install -e .[all]

4. **Install pre-commit hooks**:

   .. code-block:: bash

       pre-commit install

   Pre-commit hooks automatically enforce code quality standards before each commit.

5. **(Optional but required for HPC) Register Jupyter kernel**:

   .. code-block:: bash

       python -m ipykernel install --user --name=pvdeg-dev

   This is especially important when working on HPC systems like NREL's Kestrel.

Pre-commit Hooks
~~~~~~~~~~~~~~~~

We use pre-commit hooks to maintain code quality and consistency.

What the Hooks Do
-----------------

Our ``.pre-commit-config.yaml`` configuration runs:

* **black** - Python code formatter (line length: 88 characters)
* **flake8** - Python linter for style guide enforcement
* **jupytext** - Syncs Jupyter notebooks with Python scripts for version control
* **trailing-whitespace** - Removes trailing whitespace
* **end-of-file-fixer** - Ensures files end with newline
* **check-yaml** - Validates YAML syntax
* **check-added-large-files** - Prevents committing large files

Running Pre-commit Manually
----------------------------

Run all hooks on all files:

.. code-block:: bash

    pre-commit run --all-files

Run hooks on specific files:

.. code-block:: bash

    pre-commit run --files path/to/file.py

Skip hooks for a specific commit (use sparingly):

.. code-block:: bash

    git commit --no-verify -m "commit message"

Jupytext Synchronization
-------------------------

Notebooks in ``tutorials/`` are automatically synced with Python scripts in ``scripts/`` folders.

When you edit a notebook:

1. Pre-commit hook creates/updates a ``.py`` script version
2. Both files are staged for commit
3. Version control works well with notebooks (diffs show code changes, not JSON)

Manual sync:

.. code-block:: bash

    jupytext --sync tutorials/**/*.ipynb

Code Contributions
~~~~~~~~~~~~~~~~~~

Code Style
----------

We follow standard Python conventions:

* **PEP 8** style guide with 88 character line length (black default)
* **Google-style docstrings** for functions and classes (numpydoc format)
* Type hints encouraged for function signatures
* Descriptive variable names (avoid single letters except in loops)
* Code must be compatible with Python 3.10 and above
* Remove ``print`` statements and ``logging`` calls before committing (``warning`` is acceptable)
* Set your editor to strip trailing whitespace

**Variable naming**: PVDeg uses a mix of full and abbreviated names. Prefer full names for new contributions, especially in the API. Abbreviations can be used within functions to improve readability of formulae.

Example docstring:

.. code-block:: python

    def calculate_degradation(temp, humidity, irradiance):
        """
        Calculate degradation rate based on environmental conditions.

        Parameters
        ----------
        temp : float or array-like
            Temperature in degrees Celsius.
        humidity : float or array-like
            Relative humidity as percentage (0-100).
        irradiance : float or array-like
            Irradiance in W/m².

        Returns
        -------
        float or array-like
            Degradation rate in %/year.

        Notes
        -----
        Based on the Arrhenius equation with humidity acceleration factor.
        See [1]_ for detailed methodology.

        References
        ----------
        .. [1] Author et al. (2020). "Title." Journal. DOI.
        """
        pass

**Generic type descriptors** used in PVDeg documentation:

* **dict-like**: dict, OrderedDict, pd.Series
* **numeric**: scalar, np.array, pd.Series (typically int or float dtype)
* **array-like**: np.array, pd.Series (typically int or float dtype)

Parameters that specify a specific type require that exact input type.

Testing
-------

All code contributions should include unit tests.

**Running the test suite**:

.. code-block:: bash

    pytest tests/

**Run tests with coverage report**:

.. code-block:: bash

    pytest --cov=pvdeg --cov-report=html tests/

**Test specific modules**:

.. code-block:: bash

    pytest tests/test_spectral.py

**Test notebooks** (requires nbval):

.. code-block:: bash

    pytest --nbval tutorials/

**Writing tests**:

* Place tests in ``tests/`` directory
* Name test files ``test_<module>.py``
* Use descriptive test function names: ``test_<function>_<scenario>``
* Include edge cases and error conditions
* Use pytest fixtures for shared test data
* Mock external dependencies (weather APIs, file downloads)
* New test files must be included in ``tests/__init__.py``

**Debugging tests**: Use pytest's ``--pdb`` flag to drop into the debugger at test failures:

.. code-block:: bash

    pytest tests/test_spectral.py --pdb

This is preferred over adding ``print`` or ``logging`` statements.

Example test:

.. code-block:: python

    import pytest
    import numpy as np
    from pvdeg import spectral

    def test_spectral_degradation_basic():
        """Test spectral degradation with simple inputs."""
        wavelengths = np.array([300, 400, 500, 600, 700])
        spectrum = np.array([0.1, 0.3, 0.5, 0.4, 0.2])

        result = spectral.calc_degradation(wavelengths, spectrum)

        assert isinstance(result, float)
        assert result > 0
        assert result < 100  # Reasonable degradation range

    def test_spectral_degradation_with_nan():
        """Test spectral degradation handles NaN values."""
        wavelengths = np.array([300, 400, np.nan, 600, 700])
        spectrum = np.array([0.1, 0.3, 0.5, 0.4, 0.2])

        with pytest.raises(ValueError, match="NaN values"):
            spectral.calc_degradation(wavelengths, spectrum)

Documentation
-------------

**Building documentation locally**:

1. Install documentation dependencies:

   .. code-block:: bash

       pip install -e .[docs]

2. Build the docs:

   .. code-block:: bash

       cd docs
       make html

3. View in browser:

   .. code-block:: bash

       # Open docs/build/html/index.html

**Verifying documentation**: Read the Docs automatically builds documentation for each pull request. Confirm it renders correctly by following the ``continuous-documentation/read-the-docs`` link in the PR checks.

**Documentation types**:

* **API documentation**: Automatically generated from docstrings
* **User guide**: RST files in ``docs/source/user_guide/``
* **Tutorials**: Jupyter notebooks in ``tutorials/``
* **What's new**: Release notes in ``docs/source/whatsnew/``

**Tutorial guidelines**:

* Place tutorials in appropriate ``tutorials/<category>/`` folder
* Use descriptive names: ``01_intro_to_topic.ipynb``
* Include markdown cells explaining concepts
* Use local data files when possible (avoid API dependencies for reproducibility)
* Keep execution time under 5 minutes
* Clear outputs before committing (pre-commit will help)

Notebook Validation
-------------------

When contributing notebook changes, it's important to validate that notebooks execute correctly
and produce consistent outputs across different environments.

**Why validate notebooks?**

* Ensures reproducibility across different systems and environments
* Catches environment-specific bugs before they reach users
* Maintains consistent output formatting in version control
* Verifies that all cells execute in the correct order
* Confirms that notebooks work with current dependencies

**Using nbconvert for validation**

The ``jupyter nbconvert`` command executes notebooks in a clean kernel and validates outputs.
This is the recommended approach for testing notebook changes.

Execute a single notebook:

.. code-block:: bash

    jupyter nbconvert --to notebook --execute --inplace "tutorials/01_basics/01_basics_humidity_design.ipynb"

Execute all notebooks in a category:

.. code-block:: bash

    jupyter nbconvert --to notebook --execute --inplace "tutorials/01_basics/*.ipynb"

Execute all tutorial notebooks (use with caution - may take a long time):

.. code-block:: bash

    jupyter nbconvert --to notebook --execute --inplace "tutorials/**/*.ipynb"

**Understanding the flags**:

* ``--to notebook``: Converts back to notebook format (not HTML or other formats)
* ``--execute``: Executes all cells in order with a fresh kernel
* ``--inplace``: Overwrites the original notebook with execution results

**Using pytest with nbval**

For automated testing, use pytest with the nbval plugin to validate notebooks:

.. code-block:: bash

    # Test all notebooks
    pytest --nbval tutorials/

    # Test specific notebook
    pytest --nbval tutorials/01_basics/01_basics_humidity_design.ipynb

    # Skip output comparison (only check execution)
    pytest --nbval --nbval-lax tutorials/

The ``--nbval-lax`` flag is useful when outputs may vary slightly (e.g., timestamps, random numbers)
but you still want to verify the notebook executes without errors.

**Best practices for notebook contributions**:

1. **Clear outputs before committing**: Use ``jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace notebook.ipynb``
   or let pre-commit hooks handle it automatically

2. **Test locally first**: Always run notebooks locally before pushing to verify they work

3. **Use relative paths**: Ensure data paths work from the repository root

4. **Document dependencies**: If a notebook requires optional dependencies (e.g., PySAM),
   note this in a markdown cell at the top

5. **Handle API keys gracefully**: Use commented-out API sections with local data alternatives,
   following the pattern in existing notebooks

6. **Keep execution time reasonable**: Aim for under 5 minutes per notebook. For longer analyses,
   consider pre-computing results or creating separate demo notebooks

7. **Version control friendly**: Clear all outputs before committing (jupytext and pre-commit help with this)

**Troubleshooting notebook validation failures**:

* **Import errors**: Verify all dependencies are installed in your environment
* **Path errors**: Check that data files exist at expected locations
* **Kernel errors**: Ensure your Jupyter kernel is properly registered (see installation guide)
* **Timeout errors**: Increase timeout with ``--ExecutePreprocessor.timeout=600`` (seconds)
* **API failures**: For notebooks with API calls, use local data or mock responses in tests

Database Contributions
~~~~~~~~~~~~~~~~~~~~~~

One of the most valuable contributions you can make is adding validated material
property data and degradation parameters to our databases.

What to Contribute
------------------

We maintain several JSON databases in ``pvdeg/data/``:

* **DegradationDatabase.json**: Degradation rates, activation energies, temperature coefficients
* **H2Opermeation.json**: Water vapor transmission rates for encapsulants and barriers
* **O2permeation.json**: Oxygen permeation rates
* **AApermeation.json**: Acetic acid permeation data
* **albedo.json**: Ground albedo values for different surfaces

How to Contribute
-----------------

1. **Locate peer-reviewed data**: Published journal articles, conference proceedings, or technical reports
2. **Prepare structured data**: Follow the JSON format in existing entries
3. **Document metadata**: Include measurement conditions, equipment, uncertainties
4. **Submit via pull request**: Add your data to the appropriate JSON file with citation

Data Quality Guidelines
-----------------------

* **Peer-reviewed sources**: Prefer published research over manufacturer specs
* **Complete metadata**: Temperature, humidity, test conditions, sample details
* **Uncertainty quantification**: Include error bars or confidence intervals when available
* **Clear units**: Explicitly state all units
* **Proper citations**: Full reference with DOI when possible

Example JSON Entry
------------------

.. code-block:: json

    {
        "material": "EVA",
        "property": "activation_energy",
        "value": 0.87,
        "units": "eV",
        "conditions": {
            "temperature_range": "60-85 C",
            "test_method": "Arrhenius analysis",
            "sample_thickness": "0.46 mm"
        },
        "uncertainty": 0.05,
        "reference": {
            "authors": "Smith et al.",
            "title": "Degradation study of EVA encapsulants",
            "journal": "Solar Energy Materials",
            "year": 2020,
            "doi": "10.1016/j.solmat.2020.xxxxx"
        }
    }

Submitting Changes
~~~~~~~~~~~~~~~~~~

Pull Request Process
--------------------

1. **Create a feature branch**:

   .. code-block:: bash

       git checkout -b feature/descriptive-name

   Use prefixes: ``feature/``, ``bugfix/``, ``docs/``, ``refactor/``

2. **Make changes and commit**:

   .. code-block:: bash

       git add .
       git commit -m "Add descriptive commit message"

   Write clear commit messages explaining *why* the change was made.

3. **Push to your fork**:

   .. code-block:: bash

       git push origin feature/descriptive-name

4. **Open a pull request** on GitHub:

   * Provide a clear description of changes
   * Reference related issues (e.g., "Closes #123")
   * Ensure all CI checks pass (tests, pre-commit hooks)
   * Request review from maintainers or tag ``@NREL/pvdeg-maintainers``

**Best practices**:

* Keep pull requests focused and manageable (preferably <50 lines of primary code)
* Consider breaking large changes into smaller, incremental pull requests
* Submit pull requests early if you want feedback (mark as draft if incomplete)
* Be patient - reviewers bring diverse expertise and perspectives
* GitHub will automatically "squash and merge" your commits into a single commit

Pull Request Checklist
----------------------

Before submitting your pull request, verify:

- [ ] Code follows PEP 8 style (black formatting applied)
- [ ] All tests pass locally (``pytest tests/``)
- [ ] New functionality includes unit tests
- [ ] Docstrings added/updated for new/modified functions
- [ ] Type hints included for function signatures
- [ ] Pre-commit hooks pass (``pre-commit run --all-files``)
- [ ] Documentation updated if adding features
- [ ] What's new entry added if significant change
- [ ] Notebooks clear outputs and sync with ``.py`` scripts
- [ ] Database entries include proper citations and metadata

Code Review Process
-------------------

* Maintainers will review your pull request within a few days
* Discussion and iteration are normal - don't be discouraged
* Address feedback by pushing new commits to your branch
* Ping ``@NREL/pvdeg-maintainers`` if your pull request seems forgotten
* Once approved, maintainers will merge your changes
* Your contribution will be included in the next release

**Understanding reviews**: The PV modeling community is diverse. Reviewers may focus on:

* Algorithm details and scientific accuracy
* Integration with existing PVDeg code
* Code maintainability and style
* Test coverage and documentation quality

Contributor License Agreement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First-time contributors must sign the `Contributor License Agreement (CLA)
<https://github.com/NREL/PVDegradationTools/blob/main/cla-1.0.md>`_.
This protects both you and the project.

When you submit your first pull request, a bot will comment with instructions
to sign the CLA. Simply follow the provided link.

Release Process
~~~~~~~~~~~~~~~

For maintainers and contributors interested in the release workflow:

**Versioning**: We follow `semantic versioning <https://semver.org/>`_ (MAJOR.MINOR.PATCH)

* MAJOR: Incompatible API changes
* MINOR: New features (backward compatible)
* PATCH: Bug fixes (backward compatible)

**Testing releases**:

* Pre-release tags (``v0.7.0-rc1``, ``v0.7.0-alpha1``) → TestPyPI only
* Final tags (``v0.7.0``) → TestPyPI and PyPI

**What's new**: Update ``docs/source/whatsnew/releases/vX.Y.Z.rst`` with:

* Enhancements
* Breaking changes
* Deprecations
* Bug fixes
* Dependency updates
* Issue references (#123)

Getting Help
~~~~~~~~~~~~

If you have questions or need help:

* **Ask on GitHub Discussions**: `<https://github.com/NREL/PVDegradationTools/discussions>`_
* **Open an issue**: For bugs or feature requests
* **Check the documentation**: `<https://pvdegradationtools.readthedocs.io/>`_
* **Review existing PRs**: See how others approached similar problems

Community Guidelines
~~~~~~~~~~~~~~~~~~~~

* Be respectful and professional
* Assume positive intent in all interactions
* Focus on constructive feedback
* Welcome newcomers and help them learn
* Give credit where it's due
* Follow the `NREL Code of Conduct <https://github.com/NREL/.github/blob/main/CODE_OF_CONDUCT.md>`_

Thank you for contributing to PVDeg and supporting the PV research community! 🌞