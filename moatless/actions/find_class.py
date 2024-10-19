import logging
from typing import Optional, List

from pydantic import Field

from moatless.actions.search_base import SearchBaseAction
from moatless.index import CodeIndex
from moatless.index.types import SearchCodeResponse

logger = logging.getLogger(__name__)


class FindClass(SearchBaseAction):
    """
    Find a specific class in the codebase.
    """

    scratch_pad: str = Field(
        ..., description="Your thoughts on how to define the search."
    )

    class_name: str = Field(
        ..., description="Specific class name to include in the search."
    )

    file_pattern: Optional[str] = Field(
        default=None,
        description="A glob pattern to filter search results to specific file types or directories. ",
    )

    @property
    def log_name(self):
        return f"FindClass({self.class_name})"

    def to_prompt(self):
        prompt = f"Searching for class: {self.class_name}"
        if self.file_pattern:
            prompt += f" in files matching the pattern: {self.file_pattern}"
        return prompt

    def _search(self, code_index: CodeIndex) -> SearchCodeResponse:
        logger.info(
            f"{self.name}: {self.class_name} (file_pattern: {self.file_pattern})"
        )
        return code_index.find_class(self.class_name, file_pattern=self.file_pattern)

    def _select_span_response_prompt(self, search_result: SearchCodeResponse) -> str:
        prompt = (f"The class is too large. You must add the relevant functions to context to be able to use them. "
                    f"Use the function RequestMoreContext and specify the SpanIDs of the relevant functions to add them to context.\n")
        prompt += super()._select_span_response_prompt(search_result)
        return prompt

    def _search_for_alternative_suggestion(
        self, code_index: CodeIndex
    ) -> SearchCodeResponse:
        if self.file_pattern:
            return code_index.find_class(self.class_name, file_pattern=None)

        return SearchCodeResponse()

    def get_evaluation_criteria(self, trajectory_length) -> List[str]:
        criteria = super().get_evaluation_criteria(trajectory_length)
        criteria.extend(
            [
                "Identifier Correctness: Verify that the class name is accurate.",
            ]
        )
        return criteria
