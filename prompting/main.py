import concurrent.futures
import itertools
import json
import logging
import re
from pathlib import Path

import httpx
import tomllib
from tqdm import tqdm

from prompt_builders.xml import XMLPromptBuilder
from prompt_providers.gemini_api import GeminiAPIProvider
from prompt_providers.interface import LLMProvider
from prompt_providers.ollama import OllamaProvider
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
    # Try to find JSON within ```json ... ```
    match = re.search(r"```json\n(.*?)\n```", response_content, re.DOTALL)
    if match:
        return match.group(1)

    # Fallback to finding the first and last curly brace
    json_start = response_content.find('{')
    json_end = response_content.rfind('}') + 1

    if json_start != -1 and json_end != -1:
        return response_content[json_start:json_end]
    return response_content


def get_provider(model_name: str, provider_name: str) -> LLMProvider:
    if provider_name == "gemini":
        return GeminiAPIProvider(model_name)
    elif provider_name == "ollama":
        return OllamaProvider(model_name)
    else:
        raise ValueError(f"Unknown provider: {provider_name}")


def evaluate_batch(evaluation_provider, prompt_builder, response, batch, original_question_prompt, retries=3):
    """
    Evaluates a single batch of questions.
    """
    eval_response = None
    for _ in range(retries):
        try:
            eval_prompt = prompt_builder.build_evaluation_prompt(
                response, batch, original_question_prompt=original_question_prompt)
            eval_response = evaluation_provider.prompt(eval_prompt, chat_history=None)
            json_string = extract_json_from_response(eval_response.content)
            return json.loads(json_string)
        except json.JSONDecodeError:
            logging.warning(f"  Retrying due to JSON decoding error: {eval_response.content}")
            continue
    else:
        # All retries failed
        logging.error(f"  Failed to decode evaluation JSON after {retries} retries: {eval_response.content}")
        return {
            framework["name"]: {
                q_item: "error"
                for q in framework["questions"]
                for q_item in (q if isinstance(q, list) else [q])
            }
            for framework in batch
        }


def get_batched_evaluation(evaluation_provider, prompt_builder, response, evaluation_questions, original_question_prompt, batch_size=5, retries=3):
    """
    Gets the evaluation in batches, in parallel.
    """
    evaluation = {}
    batches = [evaluation_questions[i:i+batch_size] for i in range(0, len(evaluation_questions), batch_size)]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(evaluate_batch, evaluation_provider, prompt_builder, response, batch, original_question_prompt, retries) for batch in batches]
        for future in concurrent.futures.as_completed(futures):
            evaluation.update(future.result())

    return evaluation



def main():
    logging.info("Hello from prompting!")
    logging.info("Hello from prompting!")
    with open("prompting/configuration/models.toml", "rb") as f:
        config = tomllib.load(f)
        models = config["models"]
        paths = config.get("paths", {})
        questions_dir = Path(paths.get("questions_dir", "prompting/configuration/questions"))
        results_dir = Path(paths.get("results_dir", "results"))
        log_file = Path(paths.get("log_file", "prompting.log"))
        evaluation_model_config = next((m for m in models if m.get("use_for_evaluation")), None)

    with open("prompting/configuration/config.toml", "rb") as f:
        app_config = tomllib.load(f)
        eval_config = app_config.get("evaluation", {})
        logging_config = app_config.get("logging", {})

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format=logging_config.get("format", '%(asctime)s - %(levelname)s - %(message)s'),
        handlers=[
            logging.FileHandler(log_file),
        ],
        force=True
    )
    # Suppress verbose logging from google-generativeai
    logging.getLogger("google.generativeai").setLevel(logging_config.get("google_genai_level", "WARNING"))

    if not evaluation_model_config:
        raise ValueError("No model found for evaluation. Please set 'use_for_evaluation' to true for one of the models in models.json.")

    evaluation_provider = get_provider(evaluation_model_config["name"], evaluation_model_config["provider"])
    prompt_builder = XMLPromptBuilder()
    question_files = get_questions(questions_dir)
    model_names = [model["name"] for model in models]

    for question_path in tqdm(question_files, desc="Processing Questions"):
        question_name = question_path.stem
        question = get_question(question_path)
        for model_config in models:
            provider = get_provider(model_config["name"], model_config["provider"])
            logging.info(
                f"\n--- Using Provider: {provider.__class__.__name__} ({provider.model}) ---"
            )

            logging.info(f"\n--- Processing Question: {question_name} ---")

            existing_combinations = check_existing_results(
                results_dir, provider.model, question_name)
            possible_numbers = get_possible_numbers(question)
            keys = possible_numbers.keys()
            value_ranges = [range(1, v + 1) for v in possible_numbers.values()]
            total_combinations = 1
            for r in value_ranges:
                total_combinations *= len(r)

            for combination_values in tqdm(itertools.product(*value_ranges), total=total_combinations, desc=f"Combinations for {question_name}", leave=False):
                combination = dict(zip(keys, combination_values))

                if combination in existing_combinations:
                    logging.info(f"Skipping existing combination: {combination}")
                    continue

                logging.info(f"\n- Processing Combination: {combination}")

                full_question_parts = get_question_combination(
                    question, combination)
                prompt_text = prompt_builder.build_question_prompt(
                    full_question_parts)
                try:
                    response = provider.prompt(prompt_text, chat_history=None)
                    logging.info(f"  Response: {response.content}")
                    # Get evaluation questions
                    evaluation_questions = get_evaluation_questions(question)
                    evaluation = get_batched_evaluation(
                        evaluation_provider, prompt_builder, response.content, evaluation_questions, prompt_text,
                        batch_size=eval_config.get("batch_size", 5),
                        retries=eval_config.get("retries", 3)
                    )

                except httpx.RemoteProtocolError as e:
                    logging.error(f"  Error: {e}")
                    continue

                logging.info(f"  Evaluation: {evaluation}")

                result = Result(model_name=provider.model,
                                question_name=question_name,
                                combination=combination,
                                prompt=prompt_text,
                                response=response.content,
                                evaluation=evaluation)

                save_result(results_dir, result)
                logging.info(f"Saved result for combination: {combination}")

            logging.info(
                f"\n--- Finished processing question: {question_name} ---"
            )
        logging.info("Exporting results to CSV...")
        export_to_csv(results_dir, question_name, model_names)
        logging.info("Done.")


if __name__ == "__main__":
    main()