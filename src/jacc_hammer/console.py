"""Command line interface to Jacc-hammer."""
import click

from . import __version__


@click.command()
@click.version_option(version=__version__)
def main() -> None:
    """Jacc-hammer fuzzy matching tool."""
    pass
