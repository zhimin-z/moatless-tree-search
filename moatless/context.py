from pydantic import BaseModel

from pydantic import BaseModel

from moatless.file_context import FileContext


class Context(BaseModel):
    """
    The context of the current state.
    """

    files: FileContext
