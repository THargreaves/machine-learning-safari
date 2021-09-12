import nox  # type: ignore

nox.options.sessions = 'tests', 'lint', 'mypy'

# Parameters
versions = ['3.9']
locations = 'src', 'tests', 'noxfile.py'


@nox.session(python=versions)
def tests(session):
    args = session.posargs or ['--cov']
    session.run('poetry', 'install', external=True)
    session.run('pytest', *args)


@nox.session(python=versions)
def lint(session):
    args = session.posargs or locations
    session.run('poetry', 'install', external=True)
    session.run('flake8', *args)


@nox.session(python=versions)
def mypy(session) -> None:
    args = session.posargs or locations
    session.run('poetry', 'install', external=True)
    session.run("mypy", *args)
