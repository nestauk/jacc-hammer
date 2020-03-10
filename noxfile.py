import nox


@nox.session(python=["3.7"])
def tests(session):
    """ Example usage: nox -- tests/test_console.py """
    args = session.posargs or ["--cov"]
    session.run("poetry", "install", external=True)
    session.run("pytest", *args)
