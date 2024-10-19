import pytest
from unittest.mock import Mock, patch
from moatless.agent.agent import Agent
from moatless.workspace import Workspace


@pytest.fixture
def mock_workspace():
    return Mock(spec=Workspace)


@pytest.fixture
def mock_completion_model():
    return Mock()


def test_agent_initialization(mock_workspace, mock_completion_model):
    model_settings = Mock()

    with patch(
        "moatless.agent.agent.CompletionModel.from_settings"
    ) as mock_from_settings:
        mock_from_settings.return_value = mock_completion_model
        agent = Agent(workspace=mock_workspace, model_settings=model_settings)

    assert agent.workspace == mock_workspace
    assert agent.completion == mock_completion_model
