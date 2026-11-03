"""Eksekusi kelompok pemanggilan paralel. Thread untuk run nyata; SimGroupRunner untuk tes deterministik
(jam simulasi maju sebesar latensi maksimum kelompok, sama seperti semantik paralel)."""
from __future__ import annotations
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Sequence, TypeVar

from ..llm.mock import SimClock

T = TypeVar("T")


class ThreadGroupRunner:
    clock = staticmethod(time.perf_counter)

    def run(self, thunks: Sequence[Callable[[], T]]) -> List[T]:
        if len(thunks) == 1:
            return [thunks[0]()]
        with ThreadPoolExecutor(max_workers=len(thunks)) as ex:
            futs = [ex.submit(t) for t in thunks]
            return [f.result() for f in futs]


class SimGroupRunner:
    def __init__(self, clock: SimClock):
        self._clock = clock

    def clock(self) -> float:
        return self._clock.now()

    def run(self, thunks: Sequence[Callable[[], T]]) -> List[T]:
        out = [t() for t in thunks]                       # semua dalam grup mulai pada waktu simulasi yang sama
        self._clock.advance(max((r.latency_s for r in out), default=0.0))
        return out
