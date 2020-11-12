"""Test console module."""
import click.testing
import pytest

from jacc_hammer import console


@pytest.fixture
def runner():
    """Fixture for invoking command-line interfaces."""
    return click.testing.CliRunner()


def test_main_succeeds(runner):
    """It exists with a status code of zero."""
    result = runner.invoke(console.main)
    assert result.exit_code == 0
