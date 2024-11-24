import json
import logging
from typing import List, Type

from moatless.actions import (
    FindClass,
    FindFunction,
    FindCodeSnippet,
    SemanticSearch,
    ViewCode,
)
from moatless.actions.action import Action
from moatless.actions.append_string import AppendString
from moatless.actions.apply_change_and_test import ApplyCodeChangeAndTest
from moatless.actions.code_change import RequestCodeChange
from moatless.actions.create_file import CreateFile
from moatless.actions.edit import ClaudeEditTool
from moatless.actions.finish import Finish
from moatless.actions.insert_line import InsertLine
from moatless.actions.list_files import ListFiles
from moatless.actions.reject import Reject
from moatless.actions.run_tests import RunTests
from moatless.actions.string_replace import StringReplace
from moatless.agent.agent import ActionAgent
from moatless.agent.code_prompts import (
    CLAUDE_REACT_PROMPT,
    REACT_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    SIMPLE_CODE_PROMPT,
)
from moatless.completion.completion import (
    LLMResponseFormat,
    CompletionModel,
)
from moatless.index import CodeIndex
from moatless.message_history import MessageHistoryGenerator
from moatless.node import Node
from moatless.repository.repository import Repository
from moatless.runtime.runtime import RuntimeEnvironment
from moatless.schema import MessageHistoryType

logger = logging.getLogger(__name__)


class CodingAgent(ActionAgent):
    def generate_system_prompt(self, possible_actions: List[Type[Action]]) -> str:
        if self.system_prompt:
            prompt = self.system_prompt
        elif self.message_history_type == MessageHistoryType.REACT:
            prompt = REACT_SYSTEM_PROMPT
        else:
            prompt = SYSTEM_PROMPT

        few_shot_examples = []
        for action in possible_actions:
            examples = action.get_few_shot_examples()
            if examples:
                few_shot_examples.extend(examples)

        if few_shot_examples:
            prompt += "\n\n# Examples\nHere are some examples of how to use the available actions:\n\n"
            for i, example in enumerate(few_shot_examples):
                if self.completion.response_format == LLMResponseFormat.REACT:
                    prompt += f"\n**Example {i+1}**"
                    action_data = example.action.model_dump()
                    scratch_pad = action_data.pop("scratch_pad", "")
                    
                    # Special handling for StringReplace and CreateFile action
                    if example.action.__class__.__name__ in ["StringReplaceArgs", "CreateFileArgs", "AppendStringArgs"]:
                        prompt += f"\nTask: {example.user_input}"
                        prompt += f"\nThought: {scratch_pad}\n"
                        prompt += f"Action: {example.action.name}\n"
                        
                        if example.action.__class__.__name__ == "StringReplaceArgs":
                            prompt += f"<path>{action_data['path']}</path>\n"
                            prompt += f"<old_str>\n{action_data['old_str']}\n</old_str>\n"
                            prompt += f"<new_str>\n{action_data['new_str']}\n</new_str>\n"
                        elif example.action.__class__.__name__ == "AppendStringArgs":
                            prompt += f"<path>{action_data['path']}</path>\n"
                            prompt += f"<new_str>\n{action_data['new_str']}\n</new_str>\n"
                        elif example.action.__class__.__name__ == "CreateFileArgs":
                            prompt += f"<path>{action_data['path']}</path>\n"
                            prompt += f"<file_text>\n{action_data['file_text']}\n</file_text>\n"
                    else:
                        # Original JSON format for other actions
                        prompt += (
                            f"\nTask: {example.user_input}"
                            f"Thought: {scratch_pad}\n"
                            f"Action: {example.action.name}\n"
                            f"{json.dumps(action_data)}\n\n"
                        )
                    
                elif self.completion.response_format == LLMResponseFormat.JSON:
                    action_json = {
                        "action": example.action.model_dump(),
                        "action_type": example.action.name,
                    }
                    prompt += f"User: {example.user_input}\nAssistant:\n```json\n{json.dumps(action_json, indent=2)}\n```\n\n"

        return prompt

    def determine_possible_actions(self, node: Node) -> List[Action]:
        possible_actions = self.actions.copy()

        # Remove Finish if a sibling has already finished
        # possible_actions = self.filter_finished(node, possible_actions)

        logger.info(
            f"Possible actions for Node{node.node_id}: {[action.__class__.__name__ for action in possible_actions]}"
        )

        return possible_actions

    def filter_finished(self, node: Node, possible_actions: List[Action]):
        siblings = node.get_sibling_nodes()
        has_finished = any(child.action.name == "Finish" for child in siblings)
        if has_finished:
            possible_actions = [
                action for action in possible_actions if action.name != "Finish"
            ]
        return possible_actions

    def filter_duplicates(self, node: Node, possible_actions: List[Action]):
        # Remove actions that have been marked as duplicates
        if node.parent:
            siblings = [
                child for child in node.parent.children if child.node_id != node.node_id
            ]
            duplicate_actions = set(
                child.action.name for child in siblings if child.is_duplicate
            )
            possible_actions = [
                action
                for action in possible_actions
                if action.name not in duplicate_actions
            ]

        return possible_actions

    @classmethod
    def create(
        cls,
        repository: Repository,
        completion_model: CompletionModel,
        code_index: CodeIndex | None = None,
        runtime: RuntimeEnvironment | None = None,
        edit_completion_model: CompletionModel | None = None,
        message_history_type: MessageHistoryType = MessageHistoryType.REACT,
        **kwargs,
    ):
        system_prompt = None
        if completion_model.supports_anthropic_computer_use:
            actions = create_claude_coding_actions(
                repository=repository,
                code_index=code_index,
                completion_model=completion_model,
            )
            system_prompt = CLAUDE_REACT_PROMPT
        else:
            actions = create_edit_code_actions(
                repository=repository,
                code_index=code_index,
                completion_model=completion_model,
            )

            system_prompt = SYSTEM_PROMPT
        
        message_generator = MessageHistoryGenerator(
            message_history_type=message_history_type,
            include_file_context=True
        )

        return cls(
            completion=completion_model,
            actions=actions,
            system_prompt=system_prompt,
            message_generator=message_generator,
            **kwargs,
        )


def create_base_actions(
    repository: Repository,
    code_index: CodeIndex | None = None,
    completion_model: CompletionModel | None = None,
) -> List[Action]:
    """Create the common base actions used across all action creators."""
    return [
        SemanticSearch(
            code_index=code_index,
            repository=repository,
            completion_model=completion_model,
        ),
        FindClass(
            code_index=code_index,
            repository=repository,
            completion_model=completion_model,
        ),
        FindFunction(
            code_index=code_index,
            repository=repository,
            completion_model=completion_model,
        ),
        FindCodeSnippet(
            code_index=code_index,
            repository=repository,
            completion_model=completion_model,
        ),
        ViewCode(repository=repository),
    ]


def create_coding_actions(
    repository: Repository,
    code_index: CodeIndex | None = None,
    runtime: RuntimeEnvironment | None = None,
    identify_completion_model: CompletionModel | None = None,
    edit_completion_model: CompletionModel | None = None,
) -> List[Action]:
    actions = create_base_actions(repository, code_index, identify_completion_model)

    if runtime:
        actions.append(
            ApplyCodeChangeAndTest(
                code_index=code_index,
                repository=repository,
                runtime=runtime,
                completion_model=edit_completion_model,
            )
        )
        actions.append(
            RunTests(repository=repository, runtime=runtime, code_index=code_index)
        )
    else:
        actions.append(
            RequestCodeChange(
                repository=repository, completion_model=edit_completion_model
            )
        )

    actions.extend([Finish(), Reject()])
    return actions


def create_edit_code_actions(
    repository: Repository,
    code_index: CodeIndex | None = None,
    completion_model: CompletionModel | None = None,
) -> List[Action]:
    """Create a list of simple code modification actions."""
    actions = create_base_actions(repository, code_index, completion_model)

    edit_actions = [
        StringReplace(repository=repository, code_index=code_index),
        # InsertLine(repository=repository,  code_index=code_index),
        CreateFile(repository=repository, code_index=code_index),
        AppendString(repository=repository, code_index=code_index),
        RunTests(repository=repository, code_index=code_index),
    ]

    actions.extend(edit_actions)
    actions.extend([Finish(), Reject()])
    return actions


def create_claude_coding_actions(
    repository: Repository,
    code_index: CodeIndex | None = None,
    completion_model: CompletionModel | None = None,
) -> List[Action]:
    actions = create_base_actions(repository, code_index, completion_model)
    actions.append(
        ClaudeEditTool(code_index=code_index, repository=repository)
    )
    actions.extend([Finish(), Reject()])
    return actions


def create_all_actions(repository: Repository, code_index: CodeIndex | None = None, completion_model: CompletionModel | None = None) -> List[Action]:
    actions = create_base_actions(repository, code_index, completion_model)
    actions.extend(create_edit_code_actions(repository, code_index, completion_model))
    actions.append(ClaudeEditTool(code_index=code_index, repository=repository))
    return actions