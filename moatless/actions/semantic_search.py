from typing import Optional, List
from pydantic import Field

from moatless.actions.search_base import SearchBaseAction, logger
from moatless.index import CodeIndex
from moatless.index.types import SearchCodeResponse


class SemanticSearch(SearchBaseAction):
    """Search for code snippets based on semantic similarity."""

    scratch_pad: str = Field(
        ..., description="Your thoughts on how to define the search."
    )

    file_pattern: Optional[str] = Field(
        default=None,
        description="A glob pattern to filter search results to specific file types or directories. ",
    )

    query: str = Field(
        ...,
        description="Natural language description of what you're looking for.",
    )

    category: Optional[str] = Field(
        None,
        description="The category of files to search for. This can be 'implementation' for core implementation files or 'test' for test files.",
    )

    @property
    def log_name(self):
        return f"SemanticSearch({self.query[:20]}...)"

    def to_prompt(self):
        prompt = f"Searching for code using the query: {self.query}"
        if self.file_pattern:
            prompt += f" in files matching the pattern: {self.file_pattern}"
        return prompt

    def _search(self, code_index: CodeIndex) -> SearchCodeResponse:
        logger.info(f"{self.name}: {self.query} (file_pattern: {self.file_pattern})")

        return code_index.semantic_search(
            self.query,
            file_pattern=self.file_pattern,
            max_results=25,
            category=self.category,
        )

    def _search_for_alternative_suggestion(
        self, code_index: CodeIndex
    ) -> SearchCodeResponse:
        return SearchCodeResponse()

    def get_evaluation_criteria(self, trajectory_length) -> List[str]:
        criteria = super().get_evaluation_criteria(trajectory_length)
        criteria.extend(
            [
                "Query Relevance: Evaluate if the search query is well-defined and likely to find relevant code.",
                "Category Appropriateness: Assess if the category (implementation or test) aligns with the search intent.",
            ]
        )
        return criteria
