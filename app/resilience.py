"""Retry policy and circuit breaker."""
import random
import time
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential backoff with jitter: base * 2^(n-1), capped, +0-25% jitter."""

    max_attempts: int = 3
    base_delay: float = 0.2
    max_delay: float = 2.0
    jitter: float = 0.25

    def delay(self, failed_attempts: int, rand: Callable[[], float] = random.random) -> float:
        raw = min(self.max_delay, self.base_delay * (2 ** (failed_attempts - 1)))
        return raw * (1 + self.jitter * rand())


class CircuitBreaker:
    """Stops calling a provider that keeps failing, then probes it to recover.

    closed    -> normal; `failure_threshold` consecutive failures open the circuit
    open      -> calls are skipped until `recovery_seconds` have passed
    half_open -> exactly one probe call is allowed; success closes, failure reopens
    """

    CLOSED, OPEN, HALF_OPEN = "closed", "open", "half_open"

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_seconds: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_seconds = recovery_seconds
        self._clock = clock
        self._failures = 0
        self._opened_at = 0.0
        self._open = False
        self._probe_in_flight = False

    @property
    def state(self) -> str:
        if not self._open:
            return self.CLOSED
        if self._clock() - self._opened_at >= self.recovery_seconds:
            return self.HALF_OPEN
        return self.OPEN

    def allow(self) -> bool:
        state = self.state
        if state == self.CLOSED:
            return True
        if state == self.HALF_OPEN and not self._probe_in_flight:
            self._probe_in_flight = True
            return True
        return False

    def record_success(self) -> None:
        self._failures = 0
        self._open = False
        self._probe_in_flight = False

    def record_failure(self) -> None:
        was_half_open = self.state == self.HALF_OPEN
        self._probe_in_flight = False
        self._failures += 1
        if was_half_open or self._failures >= self.failure_threshold:
            self._open = True
            self._opened_at = self._clock()
