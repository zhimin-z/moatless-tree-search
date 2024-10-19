import json
from typing import Literal, Union

import pytest
from instructor import OpenAISchema
from pydantic import Field

from moatless.actions.action import Action, ActionExecution
from moatless.actions.finish import Finish
from moatless.actions.code_change import RequestCodeChange
from moatless.actions.request_context import RequestMoreContext
from moatless.actions.find_function import FindFunction
from moatless.actions.semantic_search import SemanticSearch


def test_create_find_function_action():
    # Prepare test data
    action_data = {
        "action_name": "FindFunction",
        "scratch_pad": "Searching for the user authentication function",
        "function_name": "authenticate_user",
        "file_pattern": "*.py",
        "class_name": "UserAuth",
    }

    # Create the action using the new create_action method
    action = Action.create_action(**action_data)

    # Verify that the correct action type was created
    assert isinstance(action, FindFunction)

    # Verify that the action properties are set correctly
    assert action.scratch_pad == "Searching for the user authentication function"
    assert action.function_name == "authenticate_user"
    assert action.file_pattern == "*.py"
    assert action.class_name == "UserAuth"

    # Test the log_name property
    assert action.log_name == "FindFunction(UserAuth.authenticate_user)"

    # Test the to_prompt method
    expected_prompt = "Searching for function: authenticate_user in class: UserAuth in files matching the pattern: *.py"
    assert action.to_prompt() == expected_prompt


def test_create_action_invalid_name():
    # Prepare test data with an invalid action name
    action_data = {"action_name": "InvalidAction", "some_field": "some_value"}

    # Verify that creating an action with an invalid name raises a ValueError
    with pytest.raises(ValueError, match="Unknown action: InvalidAction"):
        Action.create_action(**action_data)


def test_action_execution_dump_and_load():
    # Prepare test data
    action_data = {
        "action_name": "FindFunction",
        "scratch_pad": "Searching for the user authentication function",
        "function_name": "authenticate_user",
        "file_pattern": "*.py",
        "class_name": "UserAuth",
    }

    # Create the action
    action = Action.create_action(**action_data)

    # Create ActionExecution
    execution = ActionExecution(action=action)

    # Dump the execution
    dumped_execution = execution.model_dump()

    # Verify that the action_name is in the dumped data
    assert "action_name" in dumped_execution["action"]
    assert dumped_execution["action"]["action_name"] == "FindFunction"

    # Load the execution from the dumped data
    loaded_execution = ActionExecution.model_validate(dumped_execution)

    # Verify that the loaded execution has the correct action type and properties
    assert isinstance(loaded_execution.action, FindFunction)
    assert (
        loaded_execution.action.scratch_pad
        == "Searching for the user authentication function"
    )
    assert loaded_execution.action.function_name == "authenticate_user"
    assert loaded_execution.action.file_pattern == "*.py"
    assert loaded_execution.action.class_name == "UserAuth"


def test_action_schema():
    schema = RequestMoreContext.model_json_schema()
    assert "description" in schema
    assert "title" in schema


def test_take_action():
    actions = [SemanticSearch, RequestCodeChange, Finish]

    class TakeAction(OpenAISchema):
        action: Union[tuple(actions)] = Field(...)

        class Config:
            smart_union = True

    action_type = TakeAction
    schema = action_type.model_json_schema()
    assert "properties" in schema
    assert "action" in schema["properties"]
