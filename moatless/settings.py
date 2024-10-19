import enum
from dataclasses import dataclass
from typing import Optional, List

from pydantic import BaseModel, Field, model_validator


class LLMResponseFormat(enum.Enum):
    TOOLS = "tool_call"
    ANTHROPIC_TOOLS = "anthropic_tools"
    JSON = "json_mode"
    STRUCTURED_OUTPUT = "structured_output"
    TEXT = "text"


class ModelSettings(BaseModel):
    model: str = Field(
        ...,
        description="The model to use for completions.",
    )
    temperature: float = Field(
        0.0,
        description="The temperature to use for completions.",
    )
    base_url: Optional[str] = Field(
        None,
        description="The base URL for the model API.",
    )
    api_key: Optional[str] = Field(
        None,
        description="The API key for the model API.",
    )
    response_format: Optional[LLMResponseFormat] = Field(
        None,
        description="The response format for the model API.",
    )

    @model_validator(mode="before")
    def validate_response_format(cls, values):
        if "response_format" in values and values["response_format"] is not None:
            if isinstance(values["response_format"], str):
                try:
                    values["response_format"] = LLMResponseFormat(
                        values["response_format"]
                    )
                except ValueError:
                    raise ValueError(
                        f"Invalid response_format: {values['response_format']}"
                    )
        return values

    def model_dump(self, **kwargs):
        data = super().model_dump(**kwargs)
        if data["response_format"] is not None:
            data["response_format"] = data["response_format"].value
        return data

    @classmethod
    def model_validate(cls, obj):
        if (
            isinstance(obj, dict)
            and "response_format" in obj
            and obj["response_format"] is not None
        ):
            obj["response_format"] = LLMResponseFormat(obj["response_format"])
        return super().model_validate(obj)


@dataclass
class _Settings:
    # Default model used if not provided in TreeSearchSettings
    _default_model: ModelSettings | None = None

    # Cheaper model used for supporting tasks like creating commit messages
    _cheap_model: ModelSettings | None = None

    # Model used for embedding to index and search vector indexes
    _embed_model: str = "text-embedding-3-small"

    # Flag to determine if llm completions should be included when trajectories are saved
    _include_completions_in_trajectories: bool = True

    @property
    def default_model(self) -> ModelSettings:
        return self._default_model

    @default_model.setter
    def default_model(self, default_model: ModelSettings) -> None:
        self._default_model = default_model

    @property
    def cheap_model(self) -> ModelSettings | None:
        return self._cheap_model

    @cheap_model.setter
    def cheap_model(self, cheap_model: ModelSettings | None) -> None:
        self._cheap_model = cheap_model

    @property
    def embed_model(self) -> str:
        return self._embed_model

    @embed_model.setter
    def embed_model(self, embed_model: str) -> None:
        self._embed_model = embed_model


Settings = _Settings()


class TreeSearchSettings(BaseModel):
    max_expansions: int = Field(
        1,
        description="The maximum number of expansions of one state.",
    )

    max_iterations: int = Field(
        25,
        description="The maximum number of iterations to run the tree search.",
    )

    max_cost: float = Field(
        0.5,
        description="The maximum cost spent on token before finishing.",
    )

    agent_model: Optional[ModelSettings] = Field(
        default=None,
        description="The model the agent will use.",
    )

    value_function_model: Optional[ModelSettings] = Field(
        None,
        description="The model to use for building actions.",
    )

    min_finished_nodes: Optional[int] = Field(
        None,
        description="The minimum number of finished nodes to consider before finishing",
    )

    reward_threshold: Optional[int] = Field(
        100,
        description="The min reward threshold to consider before finishing.",
    )

    states_to_explore: List[str] = Field(
        ["SearchCode", "PlanToCode"],
        description="The states to explore.",
    )

    provide_feedback: bool = Field(
        False,
        description="Whether to provide feedback from previosly evaluated transitions.",
    )

    debate: bool = Field(
        False,
        description="Whether to debate the rewards to transitions.",
    )

    best_first: bool = Field(
        True,
        description="Whether to use best first search.",
    )

    exploration_constant: float = Field(
        1.41,
        description="The exploration constant for UCT.",
    )

    max_depth: int = Field(
        20,
        description="The maximum depth for one trajectory in simulations.",
    )

    # TODO: These aren't used
    progressive_widening_constant: float = Field(
        1.0,
        description="Constant for progressive widening.",
    )
    alpha: float = Field(
        0.5,
        description="Exponent for progressive widening.",
    )

    def model_dump(self, **kwargs) -> dict:
        data = super().model_dump(**kwargs)
        data["action_builder_model"] = self.agent_model.model_dump() if self.agent_model else None
        return data

    @classmethod
    def model_validate(cls, obj):
        if isinstance(obj, dict) and "action_builder_model" in obj:
            obj["action_builder_model"] = ModelSettings.model_validate(
                obj["action_builder_model"]
            )
        return super().model_validate(obj)
