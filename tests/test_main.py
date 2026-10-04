import pytest

import main


def make_cfg(tmp_path, language):
    return {
        "output_dir": tmp_path,
        "stt": {"language": language},
        "llm": {"base_url": "http://x", "model": "m"},
        "chunk_tokens": 3000,
        "overlap_sentences": 1,
        "prompts": {"translate": "t", "summary_chunk": "c", "summary_all": "a"},
    }


@pytest.fixture
def pipeline(monkeypatch, tmp_path):
    """Stub every external step; record which LLM stages ran."""
    calls = []

    class FakeClient:
        def __init__(self, *args):
            pass

        def model_available(self, model):
            return True

    monkeypatch.setattr(main.llm_mod, "OllamaClient", FakeClient)
    monkeypatch.setattr(main.ingest_mod, "probe_duration", lambda path: 60)
    monkeypatch.setattr(main.ingest_mod, "ingest", lambda src, dst, **kw: dst)
    monkeypatch.setattr(main.transcribe_mod, "transcribe",
                        lambda wav, model, **kw: [{"text": "今天上課。"}])

    def fake_translate(*args, **kw):
        calls.append("translate")
        return "Class today."

    def fake_summarize(*args, **kw):
        calls.append("summarize")
        return "# Summary"

    monkeypatch.setattr(main.llm_mod, "translate", fake_translate)
    monkeypatch.setattr(main.llm_mod, "summarize", fake_summarize)
    return calls


def test_mandarin_course_translates(pipeline, tmp_path):
    cfg = make_cfg(tmp_path, "zh")
    assert main.process_file(cfg, tmp_path / "lecture.mp3", tmp_path / "model.bin") == "done"
    assert pipeline == ["translate", "summarize"]
    out = tmp_path / "lecture"
    assert (out / "transcription.txt").read_text(encoding="utf-8") == "今天上課。"
    assert (out / "transcription_en.txt").read_text(encoding="utf-8") == "Class today."


def test_english_course_skips_translation(pipeline, tmp_path):
    cfg = make_cfg(tmp_path, "en")
    assert main.process_file(cfg, tmp_path / "lecture.mp3", tmp_path / "model.bin") == "done"
    assert pipeline == ["summarize"]
    out = tmp_path / "lecture"
    assert (out / "transcription_en.txt").read_text(encoding="utf-8") == "今天上課。"
    assert (out / "summary.md").is_file()


@pytest.mark.parametrize("flag, expected", [([], "zh"), (["--language", "en"], "en")])
def test_language_flag_overrides_config(monkeypatch, flag, expected):
    seen = {}

    def fake_preflight(cfg):
        seen["language"] = cfg["stt"]["language"]
        raise RuntimeError("stop here")

    monkeypatch.setattr(main, "preflight", fake_preflight)
    assert main.main(flag) == 1
    assert seen["language"] == expected
