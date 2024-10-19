import json
import logging
from enum import Enum
from typing import Any, Optional, Union, List

import litellm
from litellm import cost_per_token, NotFoundError
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class FileWithSpans(BaseModel):
    file_path: str = Field(
        description="The file path where the relevant code is found."
    )
    span_ids: list[str] = Field(
        default_factory=list,
        description="Span IDs identiying the relevant code spans. A span id is a unique identifier for a code sippet. It can be a class name or function name. For functions in classes separete with a dot like 'class.function'.",
    )

    def add_span_id(self, span_id):
        if span_id not in self.span_ids:
            self.span_ids.append(span_id)

    def add_span_ids(self, span_ids: list[str]):
        for span_id in span_ids:
            self.add_span_id(span_id)

    def __eq__(self, other: "FileWithSpans"):
        return self.file_path == other.file_path and self.span_ids == other.span_ids


class RankedFileSpan(BaseModel):
    file_path: str
    span_id: str
    rank: int = 0
    tokens: int = 0


class TestStatus(str, Enum):
    FAILED = "FAILED"
    PASSED = "PASSED"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"


class TestResult(BaseModel):
    status: TestStatus
    message: Optional[str] = None
    file_path: Optional[str] = None
    span_id: Optional[str] = None
    line: Optional[int] = None
    relevant_files: List[RankedFileSpan] = Field(
        default_factory=list,
        description="List of spans that are relevant to the issue",
    )


class RewardScaleEntry(BaseModel):
    min_value: int
    max_value: int
    description: str
