from unittest.mock import Mock

from moatless.actions.code_change import RequestCodeChange
from moatless.actions.find_class import FindClass
from moatless.actions.find_code_snippet import FindCodeSnippet
from moatless.actions.find_function import FindFunction
from moatless.actions.finish import Finish
from moatless.actions.reject import Reject
from moatless.actions.request_context import RequestMoreContext
from moatless.actions.run_tests import RunTests
from moatless.actions.semantic_search import SemanticSearch
from moatless.agent.code_agent import CodingAgent
from moatless.completion import CompletionModel
from moatless.file_context import FileContext
from moatless.node import Node
from moatless.repository.repository import InMemRepository
from moatless.workspace import Workspace


def test_determine_possible_actions():
    # Create a mock node with an empty file context
    node = Node.stub(file_context=FileContext(repo=InMemRepository()))
    agent = CodingAgent(workspace=Mock(Workspace), completion=Mock(CompletionModel))

    # Test 1: Basic case with empty file context
    actions = agent._determine_possible_actions(node)
    assert set(actions) == {
        SemanticSearch,
        FindClass,
        FindFunction,
        FindCodeSnippet,
        RequestMoreContext,
    }

    # Test 2: Non-empty file context without code changes
    mock_repo = InMemRepository()
    mock_repo.save_file("test.py", "hello")
    file_context = FileContext(repo=mock_repo)
    file_context.add_span_to_context("test.py", "hello")
    node = Node.stub(file_context=file_context)
    actions = agent._determine_possible_actions(node)
    assert set(actions) == {
        SemanticSearch,
        FindClass,
        FindFunction,
        FindCodeSnippet,
        RequestMoreContext,
        RequestCodeChange,
        RunTests,
    }

    # Test 3: Non-empty file context with code changes
    mock_repo = InMemRepository()
    mock_repo.save_file("test.py", "hello")
    file_context = FileContext(repo=mock_repo)
    file_context.add_span_to_context("test.py", "hello")
    file_context.get_file("test.py").apply_changes("updated")
    node = Node.stub(file_context=file_context)
    actions = agent._determine_possible_actions(node)
    assert set(actions) == {
        SemanticSearch,
        FindClass,
        FindFunction,
        FindCodeSnippet,
        RequestMoreContext,
        RequestCodeChange,
        RunTests,
        Finish,
        Reject,
    }

    # Test 4: With a finished child
    finished_child = Node.stub(action=Finish())
    node.children.append(finished_child)
    actions = agent._determine_possible_actions(node)
    assert Finish not in actions


def test_determine_possible_actions_duplicate_actions():
    node = Node(node_id=4, file_context=FileContext(repo=InMemRepository()))
    agent = CodingAgent(workspace=Mock(Workspace), completion=Mock(CompletionModel))

    duplicate_action = SemanticSearch(scratch_pad="", query="foo")
    parent = Node(node_id=1)
    first_child = Node(node_id=2, action=duplicate_action, is_duplicate=True)
    duplicate_child = Node(node_id=4, action=duplicate_action, is_duplicate=True)
    parent.children.append(first_child)
    parent.children.append(duplicate_child)
    node.parent = parent
    actions = agent._determine_possible_actions(node)
    assert SemanticSearch not in actions
