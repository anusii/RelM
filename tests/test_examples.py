import pathlib
import runpy
import pytest

path = pathlib.Path(__file__, "..", "..", "examples").resolve()
scripts = path.glob("*.py")


@pytest.mark.parametrize("script", scripts)
def test_script_execution(script):
    runpy.run_path(script)
