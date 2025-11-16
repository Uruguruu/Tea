import itertools
import random
from pathlib import Path
from typing import Any, Dict, List

from prompt_providers.interface import LLMProvider, Message
from questions import (get_frameworks, get_possible_numbers,
                       get_question, get_question_combination, get_questions)
from result_handler import (Result, check_existing_results, export_to_csv,
                            save_result)


class MockProvider(LLMProvider):
    def __init__(self, model_name: str):
        super().__init__(model_name)

    def prompt(self, user_prompt: str,
               chat_history: Any = None) -> Message:
        # Simulate a response from the model
        return Message(
            role="assistant",
            content=
            f"Mock response from {self.model} for prompt: '{user_prompt[:50]}...'"
        )


def main():
    print("Hello from prompting!")

    # Using MockProvider for demonstration
    providers: list[LLMProvider] = [
        MockProvider(model_name="mock-ollama-gemma3-12b"),
        MockProvider(model_name="mock-gemini-1.5-flash")
    ]

    question_files = get_questions()

    for provider in providers:
        print(
            f"\n--- Using Provider: {provider.__class__.__name__} ({provider.model}) ---"
        )
        for question_path in question_files:
            question_name = Path(question_path).stem
            question = get_question(question_path)
            print(f"\n--- Processing Question: {question_name} ---")

            existing_combinations = check_existing_results(
                provider.model, question_name)

            possible_numbers = get_possible_numbers(question)
            keys = possible_numbers.keys()
            value_ranges = [range(1, v + 1) for v in possible_numbers.values()]

            for combination_values in itertools.product(*value_ranges):
                combination = dict(zip(keys, combination_values))

                if combination in existing_combinations:
                    print(f"Skipping existing combination: {combination}")
                    continue

                print(f"\n- Processing Combination: {combination}")

                full_question = get_question_combination(question, combination)

                # The prompt is the second element of the list
                prompt_text = full_question[1]

                response = provider.prompt(prompt_text)

                # Get framework names for evaluation
                frameworks = get_frameworks(question)
                # Mock parsing of the response to get yes/no answers
                evaluation = {
                    framework: random.choice(["yes", "no"])
                    for framework in frameworks
                }

                result = Result(model_name=provider.model,
                                question_name=question_name,
                                combination=combination,
                                prompt=full_question,
                                response=response.content,
                                evaluation=evaluation)

                save_result(result)
                print(f"Saved result for combination: {combination}")

            print(
                f"\n--- Finished processing question: {question_name} ---")
            print("Exporting results to CSV...")
            export_to_csv(provider.model, question_name)
            print("Done.")


if __name__ == "__main__":
    main()
