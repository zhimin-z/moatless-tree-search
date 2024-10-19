import logging
from typing import List

from moatless.actions.action import Action, ActionOutput
from moatless.file_context import FileContext
from moatless.index import CodeIndex
from moatless.index.types import SearchCodeResponse
from moatless.schema import RewardScaleEntry

logger = logging.getLogger(__name__)


class SearchBaseAction(Action):
    _max_search_tokens: int = 1000
    _max_hits: int = 5

    def execute(self, file_context: FileContext | None = None) -> ActionOutput:
        if file_context is None:
            raise ValueError(
                "File context must be provided to execute the search action."
            )

        properties = {"search_hits": [], "search_tokens": 0}

        search_result = self._search(self._workspace.code_index)
        if not search_result.hits:
            alternative_suggestion = self._search_for_alternative_suggestion(
                self._workspace.code_index
            )
            if alternative_suggestion.hits:
                search_result.message += f" But found {len(alternative_suggestion.hits)} alternative suggestions."
                extra = self._select_span_response_prompt(alternative_suggestion)
                return ActionOutput(
                    message=search_result.message,
                    extra=extra,
                    properties=properties,
                    expect_correction=True,
                )
            else:
                return ActionOutput(
                    message=search_result.message, properties=properties
                )

        search_tokens = 0
        span_count = 0
        for hit in search_result.hits:
            search_tokens += sum(span.tokens for span in hit.spans)
            span_count += len(hit.spans)

            properties["search_hits"].append(
                {
                    "file_path": hit.file_path,
                    "spans": [span.span_id for span in hit.spans],
                }
            )

        properties["search_tokens"] = search_tokens

        message = search_result.message

        if search_tokens > self._max_search_tokens or span_count > self._max_hits:
            logger.info(
                f"{self.name}: Search too large. {search_tokens=} {span_count=}, will ask for clarification."
            )

            extra = self._select_span_response_prompt(search_result)
            return ActionOutput(
                message=message,
                extra=extra,
                properties=properties,
                expect_correction=True,
            )

        search_hit_str = ""
        for hit in search_result.hits:
            search_hit_str += f"\n- File: {hit.file_path}\n  Span IDs:"
            for span in hit.spans:
                file_context.add_span_to_context(hit.file_path, span.span_id)
                search_hit_str += f"\n  - {span.span_id}"

        message += f"\n{search_hit_str}"
        return ActionOutput(message=message, properties=properties)

    def _select_span_response_prompt(self, search_result: SearchCodeResponse) -> str:
        search_result_context = self._workspace.create_file_context()
        for hit in search_result.hits:
            for span in hit.spans:
                search_result_context.add_span_to_context(
                    hit.file_path, span.span_id, tokens=100
                )

        search_result_str = search_result_context.create_prompt(
            show_span_ids=True,
            show_line_numbers=False,
            exclude_comments=True,
            show_outcommented_code=True,
            outcomment_code_comment="... rest of the code",
        )

        prompt = f"\n<search_results>\n{search_result_str}\n</search_result>\n"
        return prompt

    def _search(self, code_index: CodeIndex) -> SearchCodeResponse:
        raise NotImplementedError("Subclasses must implement this method.")

    def _search_for_alternative_suggestion(
        self, code_index: CodeIndex
    ) -> SearchCodeResponse:
        return SearchCodeResponse()

    def get_evaluation_criteria(self, trajectory_length) -> List[str]:
        evaluation_criteria = super().get_evaluation_criteria(trajectory_length)
        evaluation_criteria.extend(
            [
                "Query Relevance: Evaluate if the search query or parameters are well-defined and likely to find relevant code.",
                "Search Scope Appropriateness: Check if the file patterns and class/function names narrow down the search effectively.",
                "Relevance of Search Results: Assess whether the search results are directly related to the problem and useful for making progress.",
                "Size of Search Results: Ensure that the code context provided is appropriately sized—not too large to overwhelm nor too small to be unhelpful.",
            ]
        )

        return evaluation_criteria

    def get_reward_scale(self, trajectory_length) -> List[RewardScaleEntry]:
        if trajectory_length <= 3:
            return self.generate_reward_scale_entries([
                (90, 100, "The search action is excellent, with well-defined parameters yielding only highly relevant results."),
                (75, 89, "The search action is good, with reasonable parameters yielding relevant results."),
                (25, 74, "The search action have issues with parameters or yields few or no relevant results."),
                (0, 24, "The action is counterproductive, with search results that are entirely irrelevant or excessively large, causing setbacks."),
            ])
        else:
            return self.generate_reward_scale_entries([
                (90, 100, "The search action significantly advances the solution, providing highly relevant and appropriately sized search results."),
                (75, 89, "The search action contributes positively towards solving the problem, with relevant results and minor issues."),
                (50, 74, "The search action is acceptable but may have issues with relevance or provides search results that are too large or too small."),
                (25, 49, "The search action provides results that are not helpful due to relevance or size issues."),
                (0, 24, "The search action has minimal impact, providing few relevant results."),
                (-50, -1, "The action is counterproductive, with search results that are entirely irrelevant or excessively large, causing setbacks."),
            ])
