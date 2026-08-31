from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass

PER_IP_PER_MINUTE = 5
PER_IP_PER_DAY = 40
GLOBAL_PER_DAY = 400  # under the provider cap, leaving eval headroom
DAY = 86_400
MINUTE = 60


@dataclass
class Decision:
    allowed: bool
    reason: str = ""
    retry_after: int = 0


class Limiter:
    def __init__(self) -> None:
        self._ip_minute: dict[str, deque[float]] = defaultdict(deque)
        self._ip_day: dict[str, deque[float]] = defaultdict(deque)
        self._global: deque[float] = deque()

    @staticmethod
    def _trim(window: deque[float], span: float, now: float) -> None:
        while window and now - window[0] > span:
            window.popleft()

    def check(self, ip: str) -> Decision:
        now = time.monotonic()

        self._trim(self._global, DAY, now)
        if len(self._global) >= GLOBAL_PER_DAY:
            return Decision(False, "global_daily", int(DAY - (now - self._global[0])))

        minute = self._ip_minute[ip]
        self._trim(minute, MINUTE, now)
        if len(minute) >= PER_IP_PER_MINUTE:
            return Decision(False, "ip_minute", int(MINUTE - (now - minute[0])))

        day = self._ip_day[ip]
        self._trim(day, DAY, now)
        if len(day) >= PER_IP_PER_DAY:
            return Decision(False, "ip_daily", int(DAY - (now - day[0])))

        return Decision(True)

    def record(self, ip: str) -> None:
        now = time.monotonic()
        self._ip_minute[ip].append(now)
        self._ip_day[ip].append(now)
        self._global.append(now)

    def stats(self) -> dict:
        now = time.monotonic()
        self._trim(self._global, DAY, now)
        return {
            "global_used_today": len(self._global),
            "global_cap": GLOBAL_PER_DAY,
            "unique_ips_today": sum(1 for q in self._ip_day.values() if q),
        }
