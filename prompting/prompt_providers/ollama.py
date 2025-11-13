from dataclasses import asdict
from typing import Optional, override, List

from prompt_providers.interface import LLMProvider, History, Message

from ollama import chat, ChatResponse


class OllamaProvider(LLMProvider):
    """
    Implements a formal interface for Ollama Provider
    """
    @override
    def prompt(self, user_prompt: str, chat_history: Optional[History]) -> Message:
        """
        Implements a prompt function with a chat history object and the users_prompt and returns the message.
        :param user_prompt:
        :param chat_history:
        :return:
        """
        messages: List[dict] = [
            asdict(msg)
            for msg in (chat_history or [])
        ]
        messages.append({"role": "user", "content": user_prompt})
        response: ChatResponse = chat(model=self.model, messages=messages)

        return Message(
            role=response.message["role"],
            content=response.message["content"],
        )