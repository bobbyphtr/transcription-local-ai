from scripts import transcribe


def test_whisper_runs_without_text_context(monkeypatch, tmp_path):
    commands = []

    def fake_run(cmd, timeout=None):
        commands.append(cmd)
        (tmp_path / "lecture.txt").write_text("hello", encoding="utf-8")

    monkeypatch.setattr(transcribe, "run", fake_run)
    segments = transcribe.transcribe(tmp_path / "lecture.wav", tmp_path / "model.bin", language="zh")
    cmd = commands[0]
    assert cmd[cmd.index("-mc") + 1] == "0"
    assert cmd[cmd.index("-l") + 1] == "zh"
    assert segments == [{"start": None, "end": None, "text": "hello"}]
