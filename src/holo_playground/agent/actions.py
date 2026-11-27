"""Actions a Holo agent can take, following H Company's structured-outputs format."""

from typing import Literal

from pydantic import BaseModel, Field

from holo_playground.screen import COORD_SCALE


class ClickArgs(BaseModel):
    """Click at (x, y) coordinates"""

    tool_name: Literal["click"]
    element: str
    x: int = Field(ge=0, le=COORD_SCALE, description="X coordinate as integer in [0, 1000]")
    y: int = Field(ge=0, le=COORD_SCALE, description="Y coordinate as integer in [0, 1000]")


class WriteArgs(BaseModel):
    """Type text into the currently focused element without clicking first"""

    tool_name: Literal["write"]
    content: str = Field(description="Content to write")
    press_enter: bool = Field(default=False, description="Whether to press Enter after typing")


class ScrollArgs(BaseModel):
    """Scroll the page by about three quarters of the screen"""

    tool_name: Literal["scroll"]
    direction: Literal["up", "down"] = Field(description="Direction to scroll")


class AnswerArgs(BaseModel):
    """Provide a final answer"""

    tool_name: Literal["answer"]
    content: str = Field(description="The answer content")


class Step(BaseModel):
    note: str | None = Field(
        description="Task-relevant information from the previous observation, null if nothing new."
    )
    thought: str = Field(description="Reasoning about next steps")
    tool_call: ClickArgs | WriteArgs | ScrollArgs | AnswerArgs
