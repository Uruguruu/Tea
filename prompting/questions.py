import glob
import json
import os
from typing import Any


def get_questions() -> list[str]:
    """
    Collects the file paths of all JSON prompt files located in the
    `configuration/questions` directory.

    :returns: Paths of the JSON files found in the specified directory.  Each
        element in the returned list is a string representing the file
        path, which may be absolute or relative to the project root.
    :rtype: list[str]
    """
    prompt_files_dir = "prompting/configuration/questions"
    json_files = glob.glob(os.path.join(prompt_files_dir, "*.json"))
    return json_files


def get_question(filepath: str) -> dict[str, str]:
    """
    Function to load a question from a JSON file.

    :param filepath:
        Path to the JSON file containing the question.

    :return:
        The question data as a dictionary. If the file cannot be parsed as JSON,
        an empty dictionary is returned.  If the file does not exist, the function
        prints an error message and returns `None` indirectly via the last
        attempted return.
    :rtype: dict[str, str]
    """
    try:
        with open(filepath, "r") as f:
            question = json.load(f)
    except FileNotFoundError:
        print("Question not found")
    except json.decoder.JSONDecodeError:
        print("Question not decoded correctly")
        return {}
    return question


def get_possible_numbers(question: dict[str, Any]) -> dict[str, int]:
    """
    Get the number of items in each list within the ``situation_or_context`` part of the
    question dictionary.

    :param question: The dict that holds the ``situation_or_context`` mapping among
       other keys.  The mapping contains keys whose values are iterable sequences.
    :return: A dict that maps each key from the ``situation_or_context`` mapping
       to the integer length of its sequence.
    :rtype: dict[str, int]
    """

    situation_or_context = question.get("situation_or_context", {})
    all_context_keys = situation_or_context.keys()
    lengths = {key: len(situation_or_context.get(key, [])) for key in all_context_keys}

    return lengths


def get_question_combination(question: dict[str, Any], combination: dict[str, int]) -> dict[str, Any]:
    """
    Extracts and assembles the instruction set for a question based on a
    combination mapping.
    """
    system_instructions = question.get("system_instructions", "")
    prompt = question.get("prompt", "")
    response_options = question.get("response_options", "")
    situation_or_context_obj: dict[str, list[dict[str, str]]] = question.get("situation_or_context", {})
    
    context = get_situation_or_context(situation_or_context_obj, combination)
    
    return {
        "system_instructions": system_instructions,
        "prompt": prompt,
        "context": context,
        "response_options": response_options
    }


def _get_instruction(target_object: dict[str, str]) -> str:
    """
    Gets the instruction for a target object, applying a frame template if available.
    Local frame takes precedence over global frame.
    """
    instruction_string = target_object.get("instructions", "")
    return instruction_string


def get_situation_or_context(situation_or_context: dict[str, list[dict[str, str]]], combination: dict[str, int]) -> \
list[str]:
    """
    Retrieve a list of instruction strings based on a situation/context mapping
    and a combination of keys with 1-based indices.

    A 'Frame' element can be defined globally in 'imaginary_self' or locally
    within a specific context list. The local frame will take precedence.
    """
    result: list[str] = []

    for key, index in combination.items():
        zero_based_index = index - 1
        element_list = situation_or_context.get(key, [])
        full_instruction = ""

        try:
            target_object = element_list[zero_based_index]
            full_instruction = _get_instruction(target_object)
        except (IndexError, TypeError, KeyError) as e:
            print(f"Error receiving instruction for key='{key}', index={index}: {e}")

        result.append(full_instruction)

    return result


def get_evaluation_questions(question: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Retrieves the evaluation frameworks and their questions from the question data.

    :param question: The question data.
    :return: A list of dictionaries, where each dictionary represents a framework
             and contains the framework's name and a list of questions.
    """
    frameworks = question.get("frameworks_to_decide_on")

    if not frameworks:
        raise ValueError(
            "No or faulty Framework in config. Please adjust the config!")
    if not isinstance(frameworks, list):
        raise TypeError(
            "Configuration error: 'frameworks_to_decide_on' must be a list.")

    return frameworks
