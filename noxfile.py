"""Nox sessions."""
import tempfile

import nox

locations = "src", "tests", "noxfile.py", "docs/conf.py"
nox.options.sessions = "lint", "tests"


with open(".python-version") as f:
    PYTHON_DEV_VERSION = f.read()


def install_with_constraints(session, *args, **kwargs):
    """Obey poetry constraints when installing session package."""
    with tempfile.NamedTemporaryFile() as requirements:
        session.run(
            "poetry",
            "export",
            "--dev",
            "--format=requirements.txt",
            f"--output={requirements.name}",
            external=True,
        )
        session.install(f"--constraint={requirements.name}", *args, **kwargs)


@nox.session(python=[PYTHON_DEV_VERSION])
def black(session):
    """Run black code formatter."""
    args = session.posargs or locations
    install_with_constraints(session, "black")
    session.run("black", *args)


@nox.session(python=[PYTHON_DEV_VERSION])
def lint(session):
    """Lint with flake8."""
    args = session.posargs or locations
    install_with_constraints(
        session,
        "flake8-annotations",
        "flake8-black",
        "flake8-bugbear",
        "flake8-docstrings",
        "flake8-import-order",
        "darglint",
    )
    session.run("flake8", *args)


@nox.session(python=[PYTHON_DEV_VERSION])
def tests(session):
    """Run the test suite.

    Args:
        session: Nox session

    Example usage: nox -- tests/test_console.py
    """
    args = session.posargs or ["--cov"]
    session.run("poetry", "install", "--no-dev", external=True)
    # Manually add testing dev deps
    install_with_constraints(
        session, "coverage[toml]", "pytest", "pytest-cov", "pytest-mock"
    )
    session.run("pytest", *args)


@nox.session(python=[PYTHON_DEV_VERSION])
def mypy(session):
    """Type-check with mypy."""
    args = session.posargs or locations
    install_with_constraints(session, "mypy")
    session.run("mypy", *args)


@nox.session(python=[PYTHON_DEV_VERSION])
def docs(session):
    """Build the documentation."""
    session.run("poetry", "install", "--no-dev", external=True)
    install_with_constraints(session, "sphinx", "sphinx-autodoc-typehints")
    session.run("sphinx-build", "docs", "docs/_build")
