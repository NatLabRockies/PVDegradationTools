# Contributing to PVDeg

Thank you for your interest in contributing to PVDeg! We welcome contributions from the community.

📖 **For comprehensive contributing guidelines, see the [Contributing Guide](https://pvdegradationtools.readthedocs.io/en/latest/user_guide/contributing.html) in our documentation.**

This document provides a quick reference for getting started.

## Quick Start for Developers

```bash
# 1. Fork and clone
git clone https://github.com/YOUR-USERNAME/PVDegradationTools.git
cd PVDegradationTools

# 2. Create environment (choose venv or conda)
python -m venv pvdeg-dev
source pvdeg-dev/bin/activate  # Windows: pvdeg-dev\Scripts\activate

# 3. Install in editable mode with all dependencies
pip install -e .[all]

# 4. Install pre-commit hooks
pre-commit install

# 5. (Optional, required for HPC) Register Jupyter kernel
python -m ipykernel install --user --name=pvdeg-dev
```

👉 **See detailed installation instructions, environment options, and troubleshooting in the [documentation](https://pvdegradationtools.readthedocs.io/en/latest/user_guide/contributing.html#getting-started).**

## Pre-commit Hooks

Pre-commit hooks automatically run code quality checks (black, flake8, jupytext, etc.) before each commit.

**Manual runs:**
```bash
pre-commit run --all-files              # All hooks on all files
pre-commit run --files path/to/file.py  # Specific files
```

👉 **See [Pre-commit Hooks documentation](https://pvdegradationtools.readthedocs.io/en/latest/user_guide/contributing.html#pre-commit-hooks) for details on what each hook does and jupytext synchronization.**

## Validating Notebook Outputs

When contributing notebook changes, validate outputs cleanly to avoid environment-specific formatting issues:

**Single notebook:**
```bash
jupyter nbconvert --to notebook --execute --inplace "tutorials/01_basics/01_basics_humidity_design.ipynb"
```

**All notebooks in a category:**
```bash
jupyter nbconvert --to notebook --execute --inplace "tutorials/01_basics/*.ipynb"
```

## Ways to Contribute

### Code Contributions

- Follow PEP 8 style (88 character line length)
- Include tests for new functionality
- Add/update docstrings (numpydoc format)
- Update documentation as needed

👉 **See [Code Contributions](https://pvdegradationtools.readthedocs.io/en/latest/user_guide/contributing.html#code-contributions) for detailed guidelines, examples, and testing instructions.**

### Database Contributions

PVDeg maintains community-driven open datasets for PV degradation modeling. We welcome contributions of validated material properties and degradation parameters!

**Databases in `pvdeg/data/`:**
- **DegradationDatabase.json**: Kinetic parameters, activation energies
- **H2Opermeation.json**: Water vapor permeation for encapsulants/backsheets
- **O2permeation.json**: Oxygen permeation properties
- **AApermeation.json**: Acetic acid permeation data
- **albedo.json**: UV-albedo for different surface types

**Quick guidelines:**
- Use peer-reviewed sources
- Follow existing JSON structure
- Include proper citations (DOI preferred)
- Specify units and measurement conditions
- Validate JSON syntax before submitting

👉 **See [Database Contributions](https://pvdegradationtools.readthedocs.io/en/latest/user_guide/contributing.html#database-contributions) for detailed guidelines, data quality requirements, and JSON examples.**

## Submitting Changes

```bash
# 1. Create feature branch
git checkout -b feature/my-feature

# 2. Make changes and commit (pre-commit runs automatically)
git add .
git commit -m "Description of changes"

# 3. Push and open Pull Request
git push origin feature/my-feature
```

**Pull Request Checklist:**
- [ ] Tests pass (`pytest`)
- [ ] Pre-commit hooks pass
- [ ] Documentation updated
- [ ] Changelog entry added (if user-facing change)

👉 **See [Submitting Changes](https://pvdegradationtools.readthedocs.io/en/latest/user_guide/contributing.html#submitting-changes) for the complete checklist and review process.**

## Additional Resources

- 📖 [Full Contributing Guide](https://pvdegradationtools.readthedocs.io/en/latest/user_guide/contributing.html) - Comprehensive documentation
- 📦 [Installation Guide](https://pvdegradationtools.readthedocs.io/en/latest/user_guide/installation.html) - Detailed setup instructions
- 🐛 [GitHub Issues](https://github.com/NREL/PVDegradationTools/issues) - Report bugs or request features
- 💬 [GitHub Discussions](https://github.com/NREL/PVDegradationTools/discussions) - Ask questions

## Contributor License Agreement

First-time contributors must sign the [Contributor License Agreement](cla-1.0.md). When you submit your first pull request, a bot will comment with instructions.
