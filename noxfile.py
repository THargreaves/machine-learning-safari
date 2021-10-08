"""Automation of unittests, linting, and type-checking."""
import tempfile

import nox  # type: ignore

nox.options.sessions = 'tests', 'lint', 'mypy', 'xdoctest'

# Parameters
versions = ['3.9']
locations = 'src', 'tests'


# Helper
def install_with_constraints(session, *args, **kwargs):
    """Install specific dependencies using Poetry for static analysis."""
    with tempfile.NamedTemporaryFile() as requirements:
        session.run(
            'poetry',
            'export',
            '--dev',
            '--without-hashes',
            '--format=requirements.txt',
            f"--output={requirements.name}",
            external=True,
        )
        session.install(f"--constraint={requirements.name}", *args, **kwargs)


@nox.session(python=versions)
def tests(session):
    """Perform unittests using pytest."""
    args = session.posargs or ['--cov']
    session.run('poetry', 'install', '--no-dev', external=True)
    install_with_constraints(
        session,
        'pytest',
        'coverage',
        'pytest-cov',
    )
    session.run('pytest', *args)


@nox.session(python=versions)
def lint(session):
    """Lint code using flake8."""
    args = session.posargs or locations
    install_with_constraints(
        session,
        "flake8",
        'flake8-docstrings',
        "flake8-import-order",
        'darglint',
    )
    session.run('flake8', *args)


@nox.session(python=versions)
def mypy(session) -> None:
    """Perform type-checking using mypy."""
    args = session.posargs or locations
    session.run('poetry', 'install', '--no-dev', external=True)
    install_with_constraints(session, 'mypy')
    session.run('mypy', *args)


@nox.session(python=versions)
def xdoctest(session) -> None:
    """Run examples with xdoctest."""
    args = session.posargs or ['all']
    session.run('poetry', 'install', '--no-dev', external=True)
    install_with_constraints(session, 'xdoctest', 'pygments')
    session.run('python', '-m', 'xdoctest', 'mlsafari', *args)


@nox.session(python=versions[-1])
def docs(session) -> None:
    """Build the documentation."""
    session.run('poetry', 'install', external=True)
    session.run("sphinx-build", "docs", "docs/_build")


@nox.session(python=versions[-1])
def coverage(session) -> None:
    """Upload coverage data."""
    session.run('poetry', 'install', external=True)
    session.run("coverage", "xml", "--fail-under=0")
    session.run("codecov", *session.posargs)
