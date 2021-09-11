import nox  # type: ignore

# Parameters
versions = ['3.9']
lint_locations = 'src', 'tests', 'noxfile.py'


@nox.session(python=versions)
def tests(session):
    args = session.posargs or ['--cov']
    session.run('poetry', 'install', external=True)
    session.run('pytest', *args)


@nox.session(python=versions)
def lint(session):
    args = session.posargs or lint_locations
    session.run('poetry', 'install', external=True)
    session.run('flake8', *args)
