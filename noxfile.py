"""Automation of unittests, linting, and type-checking."""
import nox  # type: ignore

nox.options.sessions = 'tests', 'lint', 'mypy'

# Parameters
versions = ['3.9']
locations = 'src', 'tests'


@nox.session(python=versions)
def tests(session):
    """Perform unittests using pytest."""
    args = session.posargs or ['--cov']
    session.run('poetry', 'install', external=True)
    session.run('pytest', *args)


@nox.session(python=versions)
def lint(session):
    """Lint code using flake8."""
    args = session.posargs or locations
    session.run('poetry', 'install', external=True)
    session.run('flake8', *args)


@nox.session(python=versions)
def mypy(session) -> None:
    """Perform type-checking using mypy."""
    args = session.posargs or locations
    session.run('poetry', 'install', external=True)
    session.run("mypy", *args)


@nox.session(python=versions)
def xdoctest(session) -> None:
    """Run examples with xdoctest."""
    args = session.posargs or ['all']
    session.run('poetry', 'install', external=True)
    session.run('python', '-m', 'xdoctest', 'machine_learning_safari', *args)


@nox.session(python=versions[-1])
def docs(session) -> None:
    """Build the documentation."""
    session.run('poetry', 'install', external=True)
    session.run("sphinx-build", "docs", "docs/_build")
