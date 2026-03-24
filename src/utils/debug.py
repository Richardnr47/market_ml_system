from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


class DebugPrinter:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def log(self, section: str, message: str) -> None:
        if self.enabled:
            print(f"[{section}] {message}")

    def line(self) -> None:
        if self.enabled:
            print("-" * 60)

    def banner(self, title: str) -> None:
        if self.enabled:
            print("\n" + "=" * 60)
            print(title)
            print("=" * 60)

    @contextmanager
    def timer(self, section: str, message: str) -> Iterator[None]:
        start = time.perf_counter()
        self.log(section, f"START: {message}")
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.log(section, f"DONE: {message} ({elapsed:.2f}s)")  