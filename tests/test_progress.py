import io

from scripts.progress import StageTimer, format_elapsed


def test_format_elapsed():
    assert format_elapsed(65) == "1:05"
    assert format_elapsed(3725) == "1:02:05"


def test_warn_prints_on_own_line():
    stream = io.StringIO()
    with StageTimer("stage", stream=stream) as stage:
        stage.warn("model reloaded")
    lines = stream.getvalue().splitlines()
    assert lines[0] == "stage  running…"
    assert lines[1] == "  ⚠ model reloaded"
    assert lines[2].startswith("stage  done in")
