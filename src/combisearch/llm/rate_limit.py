"""Simple rate limiter used to meter LLM API requests."""

from __future__ import annotations

import time


class SpeedLimitTimer:
    """Sleeps so successive ``step()`` calls are at least ``second_per_step`` apart."""

    def __init__(self, second_per_step: float = 0.75) -> None:
        self.record_time = time.time()
        self.second_per_step = second_per_step

    def step(self) -> None:
        elapsed = time.time() - self.record_time
        if elapsed <= self.second_per_step:
            time.sleep(self.second_per_step - elapsed)
        self.record_time = time.time()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)
