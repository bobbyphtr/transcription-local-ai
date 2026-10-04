"""Ollama client: chat, chunking, translation, map-reduce summarization."""
import json
import logging
import re
import urllib.request

logger = logging.getLogger("llm")

# "." only ends a sentence before whitespace or end of text, so "3.14" stays whole.
SENTENCE_END = re.compile(r"(?:[。！？!?；;\n]|\.(?=\s|$))")
# Conservative estimate: Chinese is ~1.5 chars/token (English ~4), so size
# chunks for Chinese; English chunks just come out a bit smaller.
CHARS_PER_TOKEN = 1.5

# Loop safeguard: a unit repeated LOOP_MIN_REPEATS+ times, spanning at least
# LOOP_MIN_SPAN chars and ending near the end of the reply, counts as a loop.
LOOP_MIN_REPEATS = 5
LOOP_MIN_SPAN = 200
LOOP_TAIL_CHARS = 2000
LOOP_CHECK_EVERY = 200
LOOP_PATTERN = re.compile(r"(.+?)\1{%d,}(?=.{0,%d}$)" % (LOOP_MIN_REPEATS - 1, LOOP_CHECK_EVERY), re.S)
SOURCE_LOOP_PATTERN = re.compile(r"(.+?)\1{%d,}" % (LOOP_MIN_REPEATS - 1), re.S)
MAX_LOOP_RETRIES = 2
LOOP_TEMPERATURE_STEP = 0.2


class OllamaClient:
    def __init__(self, base_url="http://localhost:11434", timeout=300):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def tags(self):
        with self._open("/api/tags") as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return [model_info.get("name", "") for model_info in data.get("models", [])]

    def model_available(self, model):
        def full_name(name):
            return name if ":" in name else name + ":latest"
        return any(full_name(name) == full_name(model) for name in self.tags())

    def chat(self, model, system, user, options=None, on_warning=None, label=""):
        """Chat with loop protection: if the reply degenerates into repetition,
        stop it, unload the model and retry with a slightly higher temperature."""
        warn = on_warning or logger.warning
        where = f" ({label})" if label else ""
        options = dict(options or {})
        for attempt in range(MAX_LOOP_RETRIES + 1):
            text, loop_at, hit_limit = self._stream_chat(model, system, user, options)
            if loop_at is None:
                if hit_limit:
                    warn(f"LLM output hit the num_predict limit{where} — this part may be cut off")
                return text.strip()
            if has_loop(user):
                warn(f"LLM output repeats because the source text repeats{where} — "
                     "keeping the output up to the loop, no reload")
                return text[:loop_at].strip()
            self.unload(model)
            if attempt < MAX_LOOP_RETRIES:
                warn(f"LLM loop detected{where} — model reloaded, retrying "
                     f"({attempt + 1}/{MAX_LOOP_RETRIES})")
                options["temperature"] = options.get("temperature", 0.2) + LOOP_TEMPERATURE_STEP
        warn(f"LLM still looping{where} after {MAX_LOOP_RETRIES} reloads — "
             "keeping the output up to the loop")
        return text[:loop_at].strip()

    def _stream_chat(self, model, system, user, options):
        """Stream one reply. Returns (text, loop_at, hit_limit); loop_at is None
        unless the reply started repeating, in which case streaming stops early.
        hit_limit is True when Ollama stopped at num_predict."""
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": True,
            # Thinking models (e.g. gemma4) otherwise spend most of num_predict on
            # hidden reasoning and the visible answer gets cut off.
            "think": False,
            "options": options,
        }
        text = ""
        checked_len = 0
        hit_limit = False
        with self._open("/api/chat", data=json.dumps(payload).encode("utf-8")) as resp:
            for line in resp:
                if not line.strip():
                    continue
                data = json.loads(line.decode("utf-8"))
                if "error" in data:
                    raise RuntimeError(f"ollama: {data['error']}")
                text += data.get("message", {}).get("content", "")
                if len(text) - checked_len >= LOOP_CHECK_EVERY:
                    checked_len = len(text)
                    loop_at = find_loop(text)
                    if loop_at is not None:
                        return text, loop_at, False  # closing resp makes Ollama stop generating
                if data.get("done"):
                    hit_limit = data.get("done_reason") == "length"
                    break
        return text, find_loop(text), hit_limit

    def unload(self, model):
        """Drop the model from memory; the next request loads it fresh."""
        payload = {"model": model, "keep_alive": 0}
        with self._open("/api/generate", data=json.dumps(payload).encode("utf-8")) as resp:
            resp.read()

    def _open(self, path, data=None):
        req = urllib.request.Request(self.base_url + path, data=data)
        req.add_header("Content-Type", "application/json")
        return urllib.request.urlopen(req, timeout=self.timeout)


def find_loop(text):
    """Return where to cut a looping reply (after the first copy of the
    repeated unit), or None if the end of the text is not a loop."""
    tail_start = max(0, len(text) - LOOP_TAIL_CHARS)
    match = LOOP_PATTERN.search(text, tail_start)
    if match is None or match.end() - match.start() < LOOP_MIN_SPAN:
        return None
    return match.end(1)


def has_loop(text):
    """True if the text contains a loop anywhere (e.g. a transcript where
    whisper repeated one line). Retrying the LLM can't fix that."""
    return any(m.end() - m.start() >= LOOP_MIN_SPAN for m in SOURCE_LOOP_PATTERN.finditer(text))


def split_sentences(text):
    pieces = re.split(f"((?:{SENTENCE_END.pattern})+)", text)
    sentences = []
    for piece_index in range(0, len(pieces), 2):
        if piece_index + 1 < len(pieces):
            sentence = (pieces[piece_index] + pieces[piece_index + 1]).strip()
        else:
            sentence = pieces[piece_index].strip()
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


def drop_first_sentences(text, count):
    sentences = split_sentences(text)
    return " ".join(sentences[count:]).strip() if count < len(sentences) else ""


def translate(client, model, prompt, chunks, overlap_sentences, options, on_progress=None, on_warning=None):
    parts = []
    total_chunks = len(chunks)
    for chunk_number, (chunk_start, chunk_end, chunk) in enumerate(chunks):
        if on_progress:
            on_progress(chunk_number + 1, total_chunks)
        out = client.chat(model, prompt, chunk, options=options, on_warning=on_warning,
                          label=f"translate chunk {chunk_number + 1}/{total_chunks}")
        if chunk_number == 0:
            parts.append(out)
        else:
            remaining = drop_first_sentences(out, overlap_sentences)
            if remaining:
                parts.append(remaining)
    return "\n\n".join(parts)


def summarize(client, model, chunk_prompt, all_prompt, chunks, options, on_progress=None, on_warning=None):
    notes = []
    total_chunks = len(chunks)
    for chunk_number, (chunk_start, chunk_end, chunk) in enumerate(chunks):
        if on_progress:
            on_progress(chunk_number + 1, total_chunks)
        notes.append(client.chat(model, chunk_prompt, chunk, options=options, on_warning=on_warning,
                                 label=f"summary chunk {chunk_number + 1}/{total_chunks}"))
    combined = "\n\n---\n\n".join(notes)
    return client.chat(model, all_prompt, combined, options=options, on_warning=on_warning,
                       label="final summary")