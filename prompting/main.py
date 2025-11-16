import itertools

from prompt_providers.interface import Message
from prompt_providers.ollama import OllamaProvider
from prompt_providers.gemini_api import GeminiAPIProvider
from questions import get_questions, get_question, get_possible_numbers, get_question_combination, get_evaluation, \
    get_frameworks
from pathlib import Path


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

    print("Hello from questions")
    question_files = get_questions()

    for question_path in question_files:
        question = get_question(question_path)
        possible_numbers = get_possible_numbers(question)
        print(f"\n--- Question: {Path(question_path).stem} ---")
        print("Possible numbers:", possible_numbers)

        keys = possible_numbers.keys()
        value_ranges = [range(1, v + 1) for v in possible_numbers.values()]

        for combination_values in itertools.product(*value_ranges):
            combination = dict(zip(keys, combination_values))
            print("\n- Combination:", combination)
            full_question = get_question_combination(question, combination)
            print("  System Instructions:", full_question[0])
            print("  Prompt:", full_question[1])
            for i, context in enumerate(full_question[2:-1]):
                print(f"  Context {i+1}:", context)
            print("  Response Options:", full_question[-1])

        print("\nEvaluation Example\n", get_evaluation(question))
        print("\n", get_frameworks(question))

if __name__ == "__main__":
    main()
