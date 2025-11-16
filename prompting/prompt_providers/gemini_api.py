from typing import Optional, override, List

from prompt_providers.interface import LLMProvider, History, Message

from google import genai
from google.genai import types

from dotenv import load_dotenv


class GeminiAPIProvider(LLMProvider):
    """
    Implements a formal interface for GeminiAPI Provider
    """
    def __init__(self, model: str):
        super().__init__(model)
        load_dotenv()
        self.client = genai.Client()

    @override
    def prompt(self, user_prompt: str, chat_history: Optional[History]) -> Message:
        """
        Implements a prompt function with a chat history object and the users_prompt and returns the message.
        :param user_prompt:
        :param chat_history:
        :return:
        """
        messages: List[Message] = (chat_history or [])

        gemini_history: List[types.Content] = [
            types.Content(
                role=map_role_to_gemini(msg.role),
                parts=[types.Part(text=msg.content)]
            )
            for msg in messages
        ]

        chat = self.client.chats.create(model=self.model, history=gemini_history)
        response = chat.send_message(user_prompt)

        return Message(role="assistant", content=response.text)

def map_role_to_gemini(role: str) -> str:
    """Konvertiert interne Rollennamen in das von Gemini erwartete Format."""
    if role == "assistant":
        return "model"
    return role