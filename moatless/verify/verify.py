from abc import ABC, abstractmethod
from typing import List

from moatless.schema import TestResult


class RuntimeEnvironment(ABC):
    @abstractmethod
    def run_tests(self, test_files: List[str] | None = None) -> list[TestResult]:
        pass
