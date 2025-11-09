from prompt.interface import Message
from prompt.ollama import OllamaProvider
from prompt.gemini_api import GeminiAPIProvider


def main():
    print("Hello from prompting!")
    test_history = [
        Message(role="user", content="Hello from the user!"),
        Message(role="assistant", content="Hello from the assistant!")
    ]

    ollama = OllamaProvider(model="gemma3:12b")
    response1 = ollama.prompt("Hello Ollama!", chat_history=None)
    response2 = ollama.prompt("What did we talk about in the last message again?", chat_history=test_history)
    print("\n------ Ollama ------\n")
    print(response1.content)
    print("\n---- Ollama with History ---\n")
    print(response2.content)

    gemini_api = GeminiAPIProvider("gemini-2.5-flash-lite")
    response1 = gemini_api.prompt("Hello Ollama!", chat_history=None)
    response2 = gemini_api.prompt("What did we talk about in the last message again?", chat_history=test_history)
    print("\n------ Gemini ------\n")
    print(response1.content)
    print("\n---- Gemini with History ---\n")
    print(response2.content)


if __name__ == "__main__":
    main()
