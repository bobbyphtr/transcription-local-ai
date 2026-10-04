from scripts import export


def test_write_output_creates_folder(tmp_path):
    path = export.write_output(tmp_path, "lecture", "transcription", "你好")
    assert path == tmp_path / "lecture" / "transcription.txt"
    assert path.read_text(encoding="utf-8") == "你好"


def test_already_done_needs_all_three(tmp_path):
    export.write_output(tmp_path, "lecture", "transcription", "t")
    export.write_output(tmp_path, "lecture", "translation", "t")
    assert not export.already_done(tmp_path, "lecture")
    export.write_output(tmp_path, "lecture", "summary", "s")
    assert export.already_done(tmp_path, "lecture")


def test_ordered_text():
    assert export.ordered_text([{"text": "a"}, {"text": "b"}]) == "a\nb"
