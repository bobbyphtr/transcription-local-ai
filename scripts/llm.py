"""Ollama client: chat, chunking, translation, map-reduce summarization."""
import json
import logging
import re
import urllib.request

logger = logging.getLogger("llm")

SENTENCE_END = re.compile(r"[。！？!?；;\n]")
CHARS_PER_TOKEN = 3.0


class OllamaClient:
    def __init__(self, base_url="http://localhost:11434", timeout=300):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def tags(self):
        with self._open("/api/tags") as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return [m.get("name", "") for m in data.get("models", [])]

    def model_available(self, model):
        prefix = model.split(":")[0]
        return any(name.split(":")[0] == prefix for name in self.tags())

    def chat(self, model, system, user, options=None):
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": options or {},
        }
        with self._open("/api/chat", data=json.dumps(payload).encode("utf-8")) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("message", {}).get("content", "").strip()

    def _open(self, path, data=None):
        req = urllib.request.Request(self.base_url + path, data=data)
        req.add_header("Content-Type", "application/json")
        return urllib.request.urlopen(req, timeout=self.timeout)


def split_sentences(text):
    parts = re.split(f"({SENTENCE_END.pattern}+)", text)
    sentences = []
    for i in range(0, len(parts), 2):
        if i + 1 < len(parts):
            sentence = (parts[i] + parts[i + 1]).strip()
        else:
            sentence = parts[i].strip()
        if sentence:
            sentences.append(sentence)
    return sentences


def chunk_text(text, max_chars, overlap_sentences=1):
    """Split text into chunks at sentence boundaries with overlap.

    Returns list of (start_index, end_index, chunk_text) into the sentence list.
    The overlap sentences are repeated at the start of the following chunk.
    """
    sentences = split_sentences(text)
    chunks = []
    start = 0
    while start < len(sentences):
        end = start
        current_len = 0
        while end < len(sentences) and current_len < max_chars:
            current_len += len(sentences[end]) + 1
            end += 1
        chunk_text = " ".join(sentences[start:end])
        chunks.append((start, end, chunk_text))
        next_start = end - overlap_sentences
        if next_start <= start:
            next_start = end
        start = next_start
    return chunks


def drop_first_sentences(text, n):
    sentences = split_sentences(text)
    return " ".join(sentences[n:]).strip() if n < len(sentences) else ""


def _as_bullets(text):
    items = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("```")]
    return "\n".join(item if item.startswith(("-", "*")) else f"- {item}" for item in items)


def translate(client, model, prompt, text, max_chars, overlap_sentences, options):
    parts = []
    for i, (_, _, chunk) in enumerate(chunk_text(text, max_chars, overlap_sentences)):
        out = client.chat(model, prompt, chunk, options=options)
        if i == 0:
            parts.append(out)
        else:
            remaining = drop_first_sentences(out, overlap_sentences)
            if remaining:
                parts.append(remaining)
    return "\n\n".join(parts)


def summarize(client, model, chunk_prompt, all_prompt, text, max_chars, options):
    bullets = [_as_bullets(client.chat(model, chunk_prompt, chunk, options=options))
               for _, _, chunk in chunk_text(text, max_chars, 0)]
    combined = "\n\n".join(bullets)
    return client.chat(model, all_prompt, combined, options=options)