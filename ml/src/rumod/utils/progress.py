"""Opt-in progress display for long-running scripts.

Library code (training, scoring, metrics) takes a `progress` flag that defaults
to False, so the API and the tests stay silent. Scripts switch it on.

Two tools:
    track(iterable, ...)  a real progress bar, for work that can be counted
    stage(label)          a live stopwatch, for a step with no measurable
                          progress (for example fitting a classifier). It shows
                          elapsed time and never fakes a percentage.

When output is not a terminal (CI logs, redirected files) stage() prints one
line when it starts and one when it ends, instead of redrawing.
"""
from __future__ import annotations

import sys
import threading
import time
from contextlib import contextmanager


def track(iterable, desc: str, total: int | None = None, enabled: bool = True):
    """Wrap an iterable in a tqdm bar, or return it untouched when disabled."""
    if not enabled:
        return iterable
    from tqdm import tqdm

    return tqdm(iterable, desc=desc, total=total, dynamic_ncols=True, mininterval=0.2)


@contextmanager
def stage(label: str, enabled: bool = True):
    """Show `label` with a live stopwatch while the block runs."""
    if not enabled:
        yield
        return

    started = time.time()

    if sys.stderr.isatty():
        from tqdm import tqdm

        bar = tqdm(total=None, desc=label, bar_format="{desc}: {elapsed}", dynamic_ncols=True)
        stop = threading.Event()

        def tick() -> None:
            while not stop.wait(0.2):
                bar.refresh()

        thread = threading.Thread(target=tick, daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join()
            bar.bar_format = "{desc}: done in {elapsed}"
            bar.refresh()
            bar.close()
    else:
        print(f"{label} ...", file=sys.stderr, flush=True)
        try:
            yield
        finally:
            print(f"{label}: done in {time.time() - started:.1f}s", file=sys.stderr, flush=True)