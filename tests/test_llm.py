from scripts import llm


class FakeClient:
    """Returns canned replies in order and records what was sent."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, model, system, user, options=None, on_warning=None, label=""):
        self.calls.append(user)
        return self.replies.pop(0)


def test_split_sentences_keeps_punctuation():
    assert llm.split_sentences("你好。今天上课！OK?") == ["你好。", "今天上课！", "OK?"]


def test_chunk_text_overlaps_and_covers_all():
    text = "一。二。三。四。五。"
    chunks = llm.chunk_text(text, max_chars=6, overlap_sentences=1)
    assert chunks[0][0] == 0
    assert chunks[-1][1] == 5
    for (_, prev_end, _), (next_start, _, _) in zip(chunks, chunks[1:]):
        assert next_start == prev_end - 1


def test_chunk_text_without_overlap():
    chunks = llm.chunk_text("一。二。三。四。", max_chars=4, overlap_sentences=0)
    assert [c[2] for c in chunks] == ["一。 二。", "三。 四。"]


def test_drop_first_sentences():
    assert llm.drop_first_sentences("A! B? C!", 1) == "B? C!"
    assert llm.drop_first_sentences("A!", 3) == ""


def test_model_available_exact_tag(monkeypatch):
    client = llm.OllamaClient()
    monkeypatch.setattr(client, "tags", lambda: ["gemma4:26b", "llama3:latest"])
    assert not client.model_available("gemma4:31b-mlx")
    assert client.model_available("gemma4:26b")
    assert client.model_available("llama3")


def test_translate_drops_overlap_sentences():
    chunks = [(0, 2, "a"), (1, 3, "b")]
    client = FakeClient(["One!\nTwo!", "Two!\nThree!"])
    out = llm.translate(client, "m", "p", chunks, 1, {})
    assert out == "One!\nTwo!\n\nThree!"


def test_summarize_joins_notes_without_bullets():
    chunks = [(0, 1, "a"), (1, 2, "b")]
    client = FakeClient(["Note one\nmore", "Note two", "FINAL"])
    out = llm.summarize(client, "m", "chunk", "all", chunks, {})
    assert out == "FINAL"
    assert client.calls[-1] == "Note one\nmore\n\n---\n\nNote two"


LECTURE = "Today we cover eigenvalues. An eigenvector keeps its direction. " * 2


def test_find_loop_ignores_normal_text():
    assert llm.find_loop(LECTURE) is None
    assert llm.find_loop("哈哈哈哈哈哈") is None
    assert llm.find_loop("Wait……………… ok.") is None


def test_find_loop_cuts_after_first_repeat():
    looping = "Intro. " + "The same sentence again. " * 30
    kept = looping[:llm.find_loop(looping)]
    assert kept.startswith("Intro. The same sentence again")
    assert kept.count("same sentence") == 1


def test_find_loop_handles_partial_last_repeat():
    looping = "Intro. " + "the same sentence again. " * 30 + "the sa"
    assert llm.find_loop(looping) is not None


def fake_ollama(monkeypatch, client, replies, done_reason="stop"):
    """Fake /api/chat streams: each reply is streamed in 10-char pieces."""
    import io
    import json
    requests = []
    replies = list(replies)

    def fake_open(path, data=None):
        payload = json.loads(data) if data else None
        requests.append((path, payload))
        if path != "/api/chat":
            return io.BytesIO(b"{}")
        text = replies.pop(0)
        lines = [json.dumps({"message": {"content": text[i:i + 10]}, "done": False})
                 for i in range(0, len(text), 10)]
        lines.append(json.dumps({"message": {"content": ""}, "done": True, "done_reason": done_reason}))
        return io.BytesIO("\n".join(lines).encode("utf-8"))

    monkeypatch.setattr(client, "_open", fake_open)
    return requests


def test_chat_plain_reply(monkeypatch):
    client = llm.OllamaClient()
    requests = fake_ollama(monkeypatch, client, [LECTURE])
    assert client.chat("m", "sys", "user", options={"temperature": 0.2}) == LECTURE.strip()
    assert [path for path, _ in requests] == ["/api/chat"]
    assert requests[0][1]["stream"] is True


def test_chat_reloads_and_retries_on_loop(monkeypatch):
    client = llm.OllamaClient()
    loop = "Intro. " + "repeat me please. " * 40
    requests = fake_ollama(monkeypatch, client, [loop, LECTURE])
    warnings = []
    out = client.chat("m", "sys", "user", options={"temperature": 0.2},
                      on_warning=warnings.append, label="chunk 1/1")
    assert out == LECTURE.strip()
    assert [path for path, _ in requests] == ["/api/chat", "/api/generate", "/api/chat"]
    assert requests[1][1] == {"model": "m", "keep_alive": 0}
    assert requests[2][1]["options"]["temperature"] == 0.2 + llm.LOOP_TEMPERATURE_STEP
    assert len(warnings) == 1 and "chunk 1/1" in warnings[0] and "reloaded" in warnings[0]


def test_chat_gives_up_and_trims_loop(monkeypatch):
    client = llm.OllamaClient()
    loop = "Intro. " + "repeat me please. " * 40
    fake_ollama(monkeypatch, client, [loop] * (llm.MAX_LOOP_RETRIES + 1))
    warnings = []
    out = client.chat("m", "sys", "user", on_warning=warnings.append)
    assert out.startswith("Intro. repeat me please")
    assert out.count("repeat") == 1
    assert len(warnings) == llm.MAX_LOOP_RETRIES + 1
    assert "still looping" in warnings[-1]


def test_split_sentences_english_periods():
    assert llm.split_sentences("Pi is 3.14 here. Next one.\nLast") == ["Pi is 3.14 here.", "Next one.", "Last"]


def test_translate_keeps_english_chunk_after_overlap():
    chunks = [(0, 2, "a"), (1, 3, "b")]
    client = FakeClient(["One. Two.", "Two. Three. Four."])
    out = llm.translate(client, "m", "p", chunks, 1, {})
    assert out == "One. Two.\n\nThree. Four."


def test_has_loop_finds_repetition_anywhere():
    assert llm.has_loop("Intro. " + "And then I can choose some lab tests.\n" * 20 + "Outro.")
    assert not llm.has_loop(LECTURE)
    assert not llm.has_loop("哈哈哈哈哈哈 ok")


def test_chat_no_reload_when_source_repeats(monkeypatch):
    client = llm.OllamaClient()
    loop = "Intro. " + "repeat me please. " * 40
    requests = fake_ollama(monkeypatch, client, [loop])
    warnings = []
    source = "Intro. " + "And then I can choose some lab tests.\n" * 20
    out = client.chat("m", "sys", source, on_warning=warnings.append)
    assert out.startswith("Intro. repeat me please") and out.count("repeat") == 1
    assert [path for path, _ in requests] == ["/api/chat"]
    assert len(warnings) == 1 and "source text repeats" in warnings[0]


def test_chat_disables_thinking(monkeypatch):
    client = llm.OllamaClient()
    requests = fake_ollama(monkeypatch, client, [LECTURE])
    client.chat("m", "sys", "user")
    assert requests[0][1]["think"] is False


def test_chat_warns_when_output_limit_hit(monkeypatch):
    client = llm.OllamaClient()
    fake_ollama(monkeypatch, client, [LECTURE], done_reason="length")
    warnings = []
    out = client.chat("m", "sys", "user", on_warning=warnings.append, label="chunk 2/3")
    assert out == LECTURE.strip()
    assert len(warnings) == 1 and "num_predict" in warnings[0] and "chunk 2/3" in warnings[0]

