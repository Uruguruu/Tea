import abc
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Message:
    """Class to represent a message from a user or assistant."""
    role: str
    content: str


type History = List[Message]


class LLMProvider(metaclass=abc.ABCMeta):
    """
    A formal interface for any class that can respond to a prompt
    with chat history.
    """
    def __init__(self, model: str):
        self.model = model

    @abc.abstractmethod
    def prompt(self, user_prompt: str, chat_history: Optional[History]) -> Message:
        """
        Takes a new user prompt and existing history,
        returns the assistant's new message.
        """
        pass

