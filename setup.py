from setuptools import setup

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

exec(open('jacc_hammer/version.py').read())

setup(
    name="jacc-hammer",
    version=__version__,
    description="Fuzzy matching",
    url="http://github.com/nestauk/jacchammer",
    author="Alex Bishop",
    author_email="alex.bishop@nesta.org.uk",
    license="MIT",
    packages=["jacc_hammer"],
    install_requires=requirements,
    python_requires=">=3.6",
)
