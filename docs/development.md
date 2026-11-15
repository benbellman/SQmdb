# SQmdb development environment

Use this environment to change SQmdb itself and run its tests. It is separate
from the Python environments used by your analyses, where you should install
a released SQmdb version from PyPI.

## Create the isolated environment

Run these commands from the repository root. Python 3.10 or newer is required.

### Windows PowerShell

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test,visualization]"
```

If PowerShell activation is unavailable in your environment, invoke the
environment's interpreter directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[test,visualization]"
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test,visualization]"
```

The `.venv/` directory is ignored by Git. It contains a local interpreter
environment and installed dependencies, not files to commit. To recreate it,
remove that specific `.venv` directory, then run the commands for your
platform again.

The `test` extra installs pytest. The `visualization` extra installs
Matplotlib and mplsoccer so the visualization tests can exercise real
rendering. Installing both extras lets you run the complete test suite.

## What an editable install means

The `-e` option installs the package in editable mode. Python imports SQmdb
directly from this repository's `sqmdb` source directory, so changes to the
source are immediately available without reinstalling after every edit. The
dependencies and package metadata are still installed into `.venv`.

This is useful for developing and testing the package, but it is not the
recommended setup for an analysis that needs to be reproducible. In an
analysis environment, install a published release from PyPI instead. That
keeps your analysis from silently using uncommitted changes in this checkout.

## Run tests and verify the imported package

With the development environment active:

```powershell
python -m pytest
```

The same command works in the macOS and Linux shells above. Confirm which
Python interpreter and SQmdb module are active with:

```powershell
python -c "import sys, sqmdb; from importlib.metadata import version; print('Python:', sys.executable); print('SQmdb version:', version('sqmdb')); print('SQmdb source:', sqmdb.__file__)"
```

For the development environment, the source path should point into this
checkout's `sqmdb` directory. The installed version comes from the
`version` field in `pyproject.toml`.

## Use a released version for an analysis

Create a different virtual environment for each analysis or reproducibility
context; do not activate this repository's `.venv` for routine analyses. In a
new analysis directory, for example:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install "sqmdb==0.1.0"
```

Replace `0.1.0` with the exact release you intend to use. The example uses
the current project version as declared in `pyproject.toml`; a version is
available from PyPI only after that release has been published. To install
the latest available release instead of pinning one, use:

```powershell
python -m pip install --upgrade sqmdb
```

Record or check the installed version with:

```powershell
python -c "from importlib.metadata import version; print(version('sqmdb'))"
```

Use the printed exact version in your analysis notes or dependency file. A
version pin identifies SQmdb, but exact environment recreation can also
depend on Python and dependency versions. The release work will address
packaging and publishing separately; this guide does not publish a package.
