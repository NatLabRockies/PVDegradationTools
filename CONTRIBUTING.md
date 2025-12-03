# Contributing to PVDeg

Thank you for your interest in contributing to PVDeg! This guide will help you get started.

## Getting Started

### Developer Installation

1. Fork and clone the repository:
   ```bash
   git clone https://github.com/YOUR-USERNAME/PVDegradationTools.git
   cd PVDegradationTools
   ```

2. Create a virtual environment:
   ```bash
   python -m venv pvdeg-dev
   source pvdeg-dev/bin/activate  # On Windows: pvdeg-dev\Scripts\activate
   ```

3. Install in editable mode with all dependencies:
   ```bash
   pip install -e .[all]
   ```

4. (Optional, but required for HPC environments) If using conda and want to use notebooks in Jupyter, register the environment as a kernel:
   ```bash
   python -m ipykernel install --user --name=pvdeg-dev
   ```
   This allows you to select the `pvdeg-dev` kernel when running Jupyter notebooks. This step is especially important when working on HPC systems like NREL's Kestrel.

### Pre-commit Hooks

We use pre-commit hooks to maintain code quality and consistency. These hooks automatically run checks before each commit.

#### Installing Pre-commit

Pre-commit is included when you install with `.[all]`, but you need to install the git hooks:

```bash
pre-commit install
```

#### What the Hooks Do

Our pre-commit configuration (`.pre-commit-config.yaml`) runs:

- **black**: Python code formatter
- **flake8**: Python linter for style guide enforcement
- **jupytext**: Syncs Jupyter notebooks with Python scripts
- **trailing-whitespace**: Removes trailing whitespace
- **end-of-file-fixer**: Ensures files end with a newline
- **check-yaml**: Validates YAML syntax
- **check-added-large-files**: Prevents committing large files

#### Running Pre-commit Manually

To run all hooks on all files:
```bash
pre-commit run --all-files
```

To run hooks on specific files:
```bash
pre-commit run --files path/to/file.py
```

To skip hooks for a specific commit (use sparingly):
```bash
git commit --no-verify -m "commit message"
```

#### Jupytext Sync

Notebooks in the `tutorials/` directory are automatically synced with Python scripts in corresponding `scripts/` folders. When you edit a notebook:

1. The pre-commit hook will create/update a `.py` script version
2. Both files will be staged for commit
3. This ensures version control works well with notebooks

To manually sync notebooks:
```bash
jupytext --sync tutorials/**/*.ipynb
```

## Making Changes

### Code Contributions

#### Code Style

- Follow PEP 8 guidelines (enforced by flake8)
- Use meaningful variable and function names
- Add docstrings to all public functions and classes
- Keep line length to 88 characters (black default)

#### Testing

Run the test suite before submitting:
```bash
pytest
```

For coverage report:
```bash
pytest --cov=pvdeg
```

#### Documentation

- Update docstrings for any changed functions
- Add examples to docstrings when helpful
- Update tutorials if adding new features
- Add entries to `docs/source/whatsnew/` for user-facing changes

### Database Contributions

PVDeg maintains community-driven open datasets for PV degradation modeling. We welcome contributions of validated material properties and degradation parameters!

#### What to Contribute

The databases are located in `pvdeg/data/` and include:

- **DegradationDatabase.json**: Kinetic parameters, activation energies, and degradation coefficients
- **H2Opermeation.json**: Water vapor permeation properties for encapsulants and backsheets
- **O2permeation.json**: Oxygen permeation properties
- **AApermeation.json**: Acetic acid permeation properties
- **albedo.json**: UV-albedo data for different surface types

#### How to Contribute Data

1. **Prepare your data**:
   - Ensure data comes from peer-reviewed publications or validated measurements
   - Include proper references (DOI, publication info)
   - Follow the existing JSON structure in the relevant database file
   - Include units and measurement conditions

2. **Add your data**:
   - Edit the appropriate JSON file in `pvdeg/data/`
   - Maintain consistent formatting and naming conventions
   - Add a comment or metadata field with the source reference

3. **Validate your changes**:
   - Ensure JSON syntax is valid (check with a JSON validator)
   - Run tests to confirm the database can still be loaded: `pytest tests/test_utilities.py`

4. **Submit a Pull Request**:
   - Include a clear description of what data you're adding
   - Provide the full citation for your data source
   - Explain any new parameters or materials being added

#### Example Database Entry

```json
{
  "material_name": {
    "parameter": value,
    "units": "unit_description",
    "source": "Author et al., Journal, Year, DOI:xxx",
    "notes": "Any relevant conditions or context"
  }
}
```

#### Data Quality Guidelines

- Data should be from published, peer-reviewed sources when possible
- Include measurement uncertainties if available
- Specify temperature, humidity, or other relevant conditions
- For kinetic parameters, include the temperature range of validity

Your contributions help build a comprehensive, validated resource for the entire PV community!

## Submitting Changes

1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/my-new-feature
   ```

2. Make your changes and commit:
   ```bash
   git add .
   git commit -m "Description of changes"
   ```
   (Pre-commit hooks will run automatically)

3. Push to your fork:
   ```bash
   git push origin feature/my-new-feature
   ```

4. Open a Pull Request on GitHub

## Contributor License Agreement

Please read and sign the Contributor License Agreement (CLA):
- Read: [cla-1.0.md](cla-1.0.md)
- Instructions: [sign-CLA.md](sign-CLA.md)

## Questions?

- Open an issue on [GitHub Issues](https://github.com/NREL/PVDegradationTools/issues)
- Check the [documentation](https://PVDegradationTools.readthedocs.io)

## Code of Conduct

Be respectful and constructive in all interactions. We aim to foster an open and welcoming environment for all contributors.
