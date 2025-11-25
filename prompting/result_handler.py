import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd


@dataclass
class Result:
    model_name: str
    question_name: str
    combination: Dict[str, int]
    prompt: str
    response: str
    evaluation: Dict[str, Dict[str, str]]


def get_results_dir(base_results_dir: Path, question_name: str, model_name: str = None) -> Path:
    """
    Retrieve (or create) the directory where results for a specific model and
    question are stored. If model_name is None, it returns the directory for the question.

    :param base_results_dir: Base directory for storing results.
    :param model_name: Name of the model whose results are being saved.
    :param question_name: Identifier for the specific question or task.
    :return: Path object pointing to the created or existing results directory.
    """
    if model_name:
        results_dir = base_results_dir / model_name / question_name
    else:
        results_dir = base_results_dir / question_name
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def save_result(base_results_dir: Path, result: Result):
    """Saves a single result to a JSON file."""
    results_dir = get_results_dir(base_results_dir, result.question_name, result.model_name)

    # Find the next available file number
    i = 0
    while True:
        result_file = results_dir / f"result_{i}.json"
        if not result_file.exists():
            break
        i += 1

    with open(result_file, "w") as f:
        json.dump(asdict(result), f, indent=4)


def load_all_results(base_results_dir: Path, question_name: str, model_name: str = None) -> List[Result]:
    """Loads all results for a given model and question."""
    if model_name:
        results_dir = get_results_dir(base_results_dir, question_name, model_name)
        results = []
        for result_file in results_dir.glob("result_*.json"):
            with open(result_file, "r") as f:
                data = json.load(f)
                results.append(Result(**data))
        return results
    else:
        results = []
        for model_dir in base_results_dir.iterdir():
            if model_dir.is_dir():
                results.extend(load_all_results(base_results_dir, question_name, model_dir.name))
        return results


def check_existing_results(base_results_dir: Path, model_name: str,
                           question_name: str) -> List[Dict[str, int]]:
    """Checks for existing results and returns a list of completed combinations."""
    results = load_all_results(base_results_dir, question_name, model_name)
    return [result.combination for result in results]


def export_to_csv(base_results_dir: Path, question_name: str, model_names: List[str]):
    """Exports all results for a given question and list of models to a CSV file."""
    all_results = []
    for model_name in model_names:
        all_results.extend(load_all_results(base_results_dir, question_name, model_name))

    if not all_results:
        return

    results_dir = get_results_dir(base_results_dir, question_name)
    csv_file = results_dir / "results.csv"

    # Flatten the data for CSV export
    flattened_data = []
    for result in all_results:
        flat_result = {
            "model_name": result.model_name,
            "question_name": result.question_name,
            "response": result.response,
        }
        # Add combination fields
        for key, value in result.combination.items():
            flat_result[f"combination_{key}"] = str(value)

        # Add evaluation fields
        for framework, questions in result.evaluation.items():
            for question, answer in questions.items():
                flat_result[f"eval_{framework}_{question}"] = answer

        # The prompt is now a single string
        flat_result["prompt"] = result.prompt

        flattened_data.append(flat_result)

    df = pd.DataFrame(flattened_data)
    df.to_csv(csv_file, index=False)
