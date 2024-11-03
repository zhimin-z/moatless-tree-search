import logging
from typing import List, Optional, Type, ClassVar

from pydantic import Field, model_validator

from moatless.actions.model import ActionArguments, FewShotExample
from moatless.actions.search_base import SearchBaseAction, SearchBaseArgs
from moatless.index.types import SearchCodeResponse

logger = logging.getLogger(__name__)


class FindCodeSnippetArgs(SearchBaseArgs):
    """Request to search for an exact code snippet."""

    code_snippet: str = Field(..., description="The exact code snippet to find.")
    file_pattern: Optional[str] = Field(
        default=None,
        description="A glob pattern to filter search results to specific file types or directories. ",
    )

    class Config:
        title = "FindCodeSnippet"

    @model_validator(mode="after")
    def validate_snippet(self) -> "FindCodeSnippetArgs":
        if not self.code_snippet.strip():
            raise ValueError("code_snippet cannot be empty")
        return self

    def to_prompt(self):
        prompt = f"Searching for code snippet: {self.code_snippet}"
        if self.file_pattern:
            prompt += f" in files matching the pattern: {self.file_pattern}"
        return prompt

class FindCodeSnippet(SearchBaseAction):
    args_schema: ClassVar[Type[ActionArguments]] = FindCodeSnippetArgs

    def _search(self, args: FindCodeSnippetArgs) -> SearchCodeResponse:
        logger.info(
            f"{self.name}: {args.code_snippet} (file_pattern: {args.file_pattern})"
        )

        return self._code_index.semantic_search(
            code_snippet=args.code_snippet,
            file_pattern=args.file_pattern,
            max_results=5,
        )

    @classmethod
    def get_few_shot_examples(cls) -> List[FewShotExample]:
        return [
            FewShotExample.create(
                user_input="Find the exact code snippet that defines the User class in our authentication module",
                response=FindCodeSnippetArgs(
                    scratch_pad="To locate the User class definition in the authentication module, we should search for the exact code snippet that declares this class.",
                    code_snippet="class User(BaseModel):"
                )
            ),
            FewShotExample.create(
                user_input="Find where we define the DEFAULT_TIMEOUT constant in our configuration files",
                response=FindCodeSnippetArgs(
                    scratch_pad="Looking for the specific line where DEFAULT_TIMEOUT is defined in configuration files.",
                    code_snippet="DEFAULT_TIMEOUT =",
                    file_pattern="**/config/*.py"
                )
            )
        ]

