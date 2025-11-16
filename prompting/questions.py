import glob
import json
import os
from typing import Any, Optional, cast


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


def get_question_combination(question: dict[str, Any], combination: dict[str, int]) -> list[str]:
    """
    Extracts and assembles the instruction set for a question based on a
    combination mapping.

    The function retrieves optional system‑level instructions and the main
    prompt from *question*, then resolves each entry defined in *combination*
    by selecting the appropriate element from the corresponding list in the
    *question*'s ``situation_or_context`` mapping.  If a ``Frame`` element is
    present in the ``imaginary_self`` list, its ``instructions`` value is used
    as a template for other instructions; otherwise the raw instruction string
    is appended.  Errors while locating an element are reported with ``print``
    and the resulting entry is left empty.

    :param question: dict[str, Any]
        Dictionary that may contain:

        * ``system_instructions`` – a list of strings with system‑level
          instructions,
        * ``prompt`` – the primary prompt string,
        * ``situation_or_context`` – a mapping where each key maps to a list of
          dictionaries, each dictionary containing at least ``name`` and
          ``instructions`` entries.
    :param combination: dict[str, int]
        Mapping from a context key to a 1‑based index indicating which element
        from the corresponding list should be used.


    :rtype: list[str]
        Ordered list consisting of the system instructions, the prompt, and
        the resolved instructions for each key in *combination*.

    :raises: None
    All lookup errors are caught internally and reported via ``print``.
    """
    system_instructions = question.get("system_instructions", [])
    prompt = question.get("prompt", "")
    response_options = question.get("response_options", [])
    situation_or_context_obj: dict[str, list[dict[str, str]]] = question.get("situation_or_context", {})
    result = [system_instructions, prompt, *get_situation_or_context(situation_or_context_obj, combination),
              response_options]
    return result


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


def get_frameworks(question: dict[str, Any]) -> list[str]:
    """
    Extract the names of frameworks from a configuration dictionary.

    The function looks for a ``frameworks_to_decide_on`` entry inside the
    provided *question* mapping.  If the entry is missing or empty,
    a :class:`ValueError` is raised.  For each framework dictionary found,
    its ``name`` field is collected.  An empty name triggers a warning
    printed to standard output, but the empty string is still added to
    the result list.

    :param question: Configuration mapping that should contain a
        ``frameworks_to_decide_on`` key.
    :type question: dict[str, Any]

    :return: List of framework names extracted from the configuration.
    :rtype: list[str]

    :raises ValueError: If ``frameworks_to_decide_on`` is missing or empty.
    """
    frameworks = question.get("frameworks_to_decide_on", {})
    if not frameworks:
        raise ValueError("No Framework for evaluation available in config. Please adjust the config!")

    result = []
    for framework in frameworks:
        framework_name = framework.get("name", "")
        result.append(framework_name)

        if not framework_name:
            print("Warning: Framework doesn't have a name.")

    return result


def get_evaluation(question: dict[str, Any]) -> list[dict[str, str]]:
    """
    Detailed summary:
    Evaluate the supplied configuration dictionary and retrieve the framework
    selection information.

    :param question: Mapping expected to contain a ``frameworks_to_decide_on`` key
        whose value holds the frameworks to be evaluated.
    :returns: The ``frameworks_to_decide_on`` sub‑dictionary extracted from the
        provided ``question`` argument.
    :raises ValueError: If the ``frameworks_to_decide_on`` entry is missing or
        evaluates to a falsy value.
    """
    frameworks = cast(list[dict[str, str]], question.get("frameworks_to_decide_on"))

    if not frameworks:
        raise ValueError("No or faulty Framework in config. Please adjust the config!")
    if not isinstance(frameworks, list):
        raise TypeError("Configuration error: 'frameworks_to_decide_on' must be a list.")

    return frameworks
