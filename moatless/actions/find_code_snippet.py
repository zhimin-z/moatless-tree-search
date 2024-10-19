from typing import Optional

from pydantic import Field

from moatless.actions.search_base import SearchBaseAction, logger
from moatless.index import CodeIndex
from moatless.index.types import SearchCodeResponse


class FindCodeSnippet(SearchBaseAction):
    """
    Request to search for an exact code snippet.
    """

    scratch_pad: str = Field(
        ..., description="Your thoughts on how to define the search."
    )

    file_pattern: Optional[str] = Field(
        default=None,
        description="A glob pattern to filter search results to specific file types or directories. ",
    )

    code_snippet: str = Field(
        ...,
        description="The exact code snippet to find.",
    )

    @property
    def log_name(self):
        return f"FindCodeSnippet({self.code_snippet[:20]}...)"

    def to_prompt(self):
        prompt = f"Searching for code snippet: {self.code_snippet}"
        if self.file_pattern:
            prompt += f" in files matching the pattern: {self.file_pattern}"
        return prompt

    def _search(self, code_index: CodeIndex) -> SearchCodeResponse:
        logger.info(
            f"{self.name}: {self.code_snippet} (file_pattern: {self.file_pattern})"
        )

        return code_index.semantic_search(
            code_snippet=self.code_snippet,
            file_pattern=self.file_pattern,
            max_results=5,
        )
