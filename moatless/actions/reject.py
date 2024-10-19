from pydantic import model_validator, Field
from moatless.actions.action import Action, ActionOutput
from moatless.file_context import FileContext


class Reject(Action):
    """Reject the task and explain why."""

    scratch_pad: str = Field(..., description="Your reasoning.")

    rejection_reason: str = Field(..., description="Explanation for rejection.")

    @model_validator(mode="before")
    def convert_reason(cls, values):
        if values and isinstance(values, dict):
            if values.get("reason"):
                values["rejection_reason"] = values["reason"]

            if "rejection_reason" not in values:
                values["rejection_reason"] = "No reason given."

            if "scratch_pad" not in values:
                values["scratch_pad"] = ""

            return values

    def to_prompt(self):
        return f"Reject with reason: {self.rejection_reason}"

    def execute(self, file_context: FileContext | None = None):
        return ActionOutput(message=self.rejection_reason, terminal=True)
