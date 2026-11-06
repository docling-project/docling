# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""The timed text that every ASR backend returns."""

from pydantic import BaseModel, Field


class _ConversationWord(BaseModel):
    text: str
    start_time: float | None = Field(
        None, description="Start time in seconds from video start"
    )
    end_time: float | None = Field(
        None, ge=0, description="End time in seconds from video start"
    )


class _ConversationItem(BaseModel):
    text: str
    start_time: float | None = Field(
        None, description="Start time in seconds from video start"
    )
    end_time: float | None = Field(
        None, ge=0, description="End time in seconds from video start"
    )
    speaker_id: int | None = Field(None, description="Numeric speaker identifier")
    speaker: str | None = Field(
        None, description="Speaker name, defaults to speaker-{speaker_id}"
    )
    words: list[_ConversationWord] | None = Field(
        None, description="Individual words with time-stamps"
    )

    def __lt__(self, other):
        if not isinstance(other, _ConversationItem):
            return NotImplemented
        return self.start_time < other.start_time

    def __eq__(self, other):
        if not isinstance(other, _ConversationItem):
            return NotImplemented
        return self.start_time == other.start_time

    def to_string(self) -> str:
        """Format the conversation entry as a string"""
        result = ""
        if (self.start_time is not None) and (self.end_time is not None):
            result += f"[time: {self.start_time}-{self.end_time}] "

        if self.speaker is not None:
            result += f"[speaker:{self.speaker}] "

        result += self.text
        return result
