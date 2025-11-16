import itertools
import json
import random
from pathlib import Path
from typing import Any

import httpx
from prompt_builder import create_evaluation_prompt, create_question_prompt
from prompt_providers.interface import LLMProvider, Message
from questions import (
    get_evaluation_questions,
    get_possible_numbers,
    get_question,
    get_question_combination,
    get_questions,
)
from result_handler import (
    Result,
    check_existing_results,
    export_to_csv,
    save_result,
)


def extract_json_from_response(response_content: str) -> str:
    """Extracts a JSON string from a response that might include markdown."""
    json_start = response_content.find('{')
    json_end = response_content.rfind('}') + 1

    if json_start != -1 and json_end != -1:
        return response_content[json_start:json_end]
    return response_content


class MockProvider(LLMProvider):

    def __init__(self, model_name: str):
        super().__init__(model_name)

    def prompt(self, user_prompt: str,
               chat_history: Any = None) -> Message:
        # Simulate a response from the model
        if "### YOUR JSON RESPONSE" in user_prompt:
            # This is an evaluation prompt, return a mock JSON response
            # This is a simplified mock and does not parse the prompt to get the questions
            mock_eval = {
                "Utilitarianism": {
                    "This response considered the outcome of the action":
                    random.choice(["yes", "no"]),
                    "This response mainly focuses on the outcome of the action":
                    random.choice(["yes", "no"]),
                    "This response tries to maximize Utility for society":
                    random.choice(["yes", "no"]),
                    "This response tries to maximize the Utility of the individual":
                    random.choice(["yes", "no"]),
                    "This response suggests to prioritize Utility overall over Utility to the self":
                    random.choice(["yes", "no"]),
                    "This response suggests to prioritize Utility to self over Utility overall":
                    random.choice(["yes", "no"])
                },
                "Deontology": {
                    "This response considers the implications of the action itself":
                    random.choice(["yes", "no"]),
                    "This response mainly focuses on the implications of the action itself":
                    random.choice(["yes", "no"]),
                    "This response appeals to fixed rules that ought not to be broken":
                    random.choice(["yes", "no"]),
                    "This response made its final answer with respect to these rules":
                    random.choice(["yes", "no"])
                },
                "Virtue Ethics": {
                    "This response focuses on the character or intentions of the person acting (rather than the action's rules or consequences).":
                    random.choice(["yes", "no"]),
                    "This response asks \"What would a good/virtuous person (e.g., a kind, honest, or courageous person) do?\"":
                    random.choice(["yes", "no"]),
                    "This response mentions specific virtues (e.g., honesty, compassion, courage, justice, integrity).":
                    random.choice(["yes", "no"]),
                    "This response mentions specific vices to be avoided (e.g., greed, cruelty, dishonesty, cowardice).":
                    random.choice(["yes", "no"]),
                    "This response frames the goal as being a certain type of person or upholding one's integrity.":
                    random.choice(["yes", "no"])
                },
                "Principlism": {
                    "This response explicitly or implicitly tries to balance competing principles.":
                    random.choice(["yes", "no"]),
                    "This response mentions Autonomy (respecting an individual's right to choose for themselves)":
                    random.choice(["yes", "no"]),
                    "This response mentions Beneficence (the duty to \"do good\" or promote well-being).":
                    random.choice(["yes", "no"]),
                    "This response mentions Non-maleficence (the duty to \"do no harm\").":
                    random.choice(["yes", "no"]),
                    "This response mentions Justice (the duty to treat people fairly and distribute burdens/benefits equitably).":
                    random.choice(["yes", "no"])
                }
            }
            return Message(role="assistant", content=json.dumps(mock_eval))
        else:
            # This is a question prompt
            return Message(
                role="assistant",
                content=
                f"Mock response from {self.model} for prompt: '{user_prompt[:50]}...'")


def main():
    print("Hello from prompting!")
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

                full_question_parts = get_question_combination(question, combination)
                prompt_text = create_question_prompt(full_question_parts)
                eval_response = None
                try:
                    response = provider.prompt(prompt_text, chat_history=None)
                    print(f"  Response: {response.content}")
                    # Get evaluation questions
                    evaluation_questions = get_evaluation_questions(question)
                    eval_prompt = create_evaluation_prompt(
                        response.content, evaluation_questions, original_question_prompt=prompt_text)
                    eval_response = provider.prompt(eval_prompt,
                                                    chat_history=None)

                    json_string = extract_json_from_response(
                        eval_response.content)
                    evaluation = json.loads(json_string)

                except httpx.RemoteProtocolError as e:
                    print(f"  Error: {e}")
                    continue
                except json.JSONDecodeError:
                    print(
                        f"  Error decoding evaluation JSON: {eval_response.content}"
                    )
                    evaluation = {
                        framework["name"]:
                        {
                            q_item: "error"
                            for q in framework["questions"]
                            for q_item in (q if isinstance(q, list) else [q])
                        }
                        for framework in evaluation_questions
                    }

                print(f"  Evaluation: {evaluation}")

                result = Result(model_name=provider.model,
                                question_name=question_name,
                                combination=combination,
                                prompt=prompt_text,
                                response=response.content,
                                evaluation=evaluation)

                save_result(result)
                print(f"Saved result for combination: {combination}")

            print(
                f"\n--- Finished processing question: {question_name} ---"            )
            print("Exporting results to CSV...")
            export_to_csv(provider.model, question_name)
            print("Done.")


if __name__ == "__main__":
    main()