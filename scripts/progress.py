"""Terminal progress: a labelled stage with a live elapsed timer.

Honest by design: whisper-cli does not expose a parseable percentage, so
instead of a fake progress bar we show (a) which stage is running and
(b) a live mm:ss timer. For the LLM steps the number of chunks is known
up front, so we show a real chunk counter (e.g. "chunk 3/7").

In non-interactive runs (logs, pipes) the live timer is turned off and each
stage prints one concise line, so captured output stays clean:

    [whisper] lecture  · running…
    [whisper] lecture  · done in 2:41
"""
import sys
import threading
import time

ANSI_ERASE_LINE = "\x1b[K"


def format_elapsed(seconds):
    """Render a duration as mm:ss or h:mm:ss."""
    seconds = int(seconds)
    minutes, remaining_seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{remaining_seconds:02d}"
    return f"{minutes}:{remaining_seconds:02d}"


class StageTimer:
    """Show a stage label, then a live elapsed timer while work runs.

    with StageTimer(f"[{index}/{total}] {name}  whisper") as stage:
        do_work()
        stage.set_progress(done=3, total=7)   # optional chunk counter

    On exit it prints '<label>  done in mm:ss' (or 'failed in mm:ss').
    """

    def __init__(self, label, stream=None):
        self.label = label
        self.stream = stream if stream is not None else sys.stderr
        self.live = self.stream.isatty()
        self._stop_event = threading.Event()
        self._tick_thread = None
        self._started_at = 0.0
        self._done = 0
        self._total = 0
        self._note = ""
        self._state_lock = threading.Lock()

    def __enter__(self):
        self._started_at = time.monotonic()
        self.stream.write(f"{self.label}  running…\n")
        self.stream.flush()
        if self.live:
            self._tick_thread = threading.Thread(target=self._tick, daemon=True)
            self._tick_thread.start()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self._stop_event.set()
        if self._tick_thread:
            self._tick_thread.join(timeout=1.0)
        if self.live:
            self.stream.write("\r" + ANSI_ERASE_LINE)
        outcome = "failed" if exc_type else "done"
        elapsed = time.monotonic() - self._started_at
        self.stream.write(f"{self.label}  {outcome} in {format_elapsed(elapsed)}\n")
        self.stream.flush()
        return False

    def set_progress(self, done, total=0, note=""):
        """Update what the live line shows while the stage is busy."""
        with self._state_lock:
            self._done = done
            self._total = total
            self._note = note

    def _tick(self):
        while not self._stop_event.wait(0.2):
            with self._state_lock:
                done, total, note = self._done, self._total, self._note
            suffix = ""
            if total:
                suffix = f"  ({done}/{total} chunks)"
            elif note:
                suffix = f"  {note}"
            elapsed = time.monotonic() - self._started_at
            line = f"\r{self.label}  elapsed {format_elapsed(elapsed)}{suffix}"
            self.stream.write(line + ANSI_ERASE_LINE)
            self.stream.flush()