from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator
import logging


class DebugPrinter:
    def __init__(self, logger: logging.Logger, enabled: bool = True) -> None:
        self.logger = logger
        self.enabled = enabled

    def log(self, section: str, message: str) -> None:
        if self.enabled:
            self.logger.info(f"[{section}] {message}")

    def debug(self, section: str, message: str) -> None:
        if self.enabled:
            self.logger.debug(f"[{section}] {message}")

    def warning(self, section: str, message: str) -> None:
        self.logger.warning(f"[{section}] {message}")

    def error(self, section: str, message: str) -> None:
        self.logger.error(f"[{section}] {message}")

    def line(self) -> None:
        if self.enabled:
            self.logger.info("-" * 80)

    def banner(self, title: str) -> None:
        if self.enabled:
            self.logger.info("=" * 80)
            self.logger.info(title)
            self.logger.info("=" * 80)

    @contextmanager
    def timer(self, section: str, message: str) -> Iterator[None]:
        start = time.perf_counter()
        self.log(section, f"START: {message}")
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.log(section, f"DONE: {message} ({elapsed:.2f}s)")