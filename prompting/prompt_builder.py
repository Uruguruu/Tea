from typing import Any

def create_question_prompt(question_parts: dict[str, Any]) -> str:
    """
    Create a formatted prompt string based on the supplied question parts.

    The function extracts optional elements from the ``question_parts`` mapping –
    system instructions, the main task prompt, a list of context strings, and
    response options – and assembles them into a markdown‑style prompt.  Empty or
    missing entries are ignored, and the resulting sections are separated by
    blank lines.

    Parameters
    ----------
    question_parts : dict[str, Any]
        Mapping that may contain the keys ``system_instructions``, ``prompt``,
        ``context``, and ``response_options``.  The values are expected to be a
        string for the first three keys and a list of strings for ``context``.

    Returns
    -------
    str
        The concatenated prompt with sections ``### System Instructions``,
        ``### Task``, ``### Context`` (if any non‑empty items are present), and
        ``### Response Options`` (if provided).  Sections are separated by two
        line breaks.

    Raises
    ------
    None
        The implementation does not raise any custom exceptions; missing keys
        are treated as empty values.

    Notes
    -----
    * The ``context`` list is filtered to discard falsy entries before being
      joined with newlines.
    * The ordering of sections is fixed: system instructions, task, context,
      response options.
    """
    system_instructions = question_parts.get("system_instructions", "")
    prompt = question_parts.get("prompt", "")
    context = question_parts.get("context", [])
    response_options = question_parts.get("response_options", "")

    prompt_parts = []
    if system_instructions:
        prompt_parts.append("### System Instructions\n" + system_instructions)
    
    prompt_parts.append("### Task\n" + prompt)

    if context:
        context = [c for c in context if c]
        if context:
            prompt_parts.append("### Context\n" + "\n".join(context))

    if response_options:
        prompt_parts.append("### Response Options\n" + response_options)

    return "\n\n".join(prompt_parts)


def _build_json_skeleton(evaluation_questions: list[dict[str, Any]]) -> list[str]:
    """
    Build a JSON‑compatible skeleton from a collection of evaluation questions.

    The function processes each framework entry supplied in *evaluation_questions*,
    extracts the framework's name and associated questions, and assembles a list of
    strings that represent partial JSON objects.  Each question may be a single
    string or a list of sub‑questions; every sub‑question is mapped to the placeholder
    value ``"yes_or_no"`` in the resulting JSON fragment.

    :param evaluation_questions: A list of dictionaries, each describing a
        framework.  Every dictionary must contain a ``"name"`` key with the
        framework identifier and a ``"questions"`` key with a collection of
        question entries.  A question entry can be a string or a list of
        strings representing sub‑questions.
    :return: A list of strings, each string being a partial JSON object for a
        framework, ready to be concatenated into a full JSON document.
    :raises: No exceptions are raised by this function.
    """
    framework_json_parts = []
    for framework in evaluation_questions:
        framework_name = framework["name"]
        questions = framework["questions"]
        question_lines = [
            f'      "{sub_q}": "yes_or_no"'
            for q in questions
            for sub_q in (q if isinstance(q, list) else [q])
        ]
        framework_json_parts.append(
            f'  "{framework_name}": {{\n' + ",\n".join(question_lines) + "\n  }"
        )
    return framework_json_parts


def create_evaluation_prompt(
        response: str,
        evaluation_questions: list[dict[str, Any]],
        original_question_prompt: str,
) -> str:
    """
    Detailed summary:
    Generate a complete evaluation prompt for an LLM‑based evaluator. The prompt
    contains clear instructions, the original question, the response to be
    evaluated, and a JSON skeleton derived from the supplied evaluation
    frameworks. The resulting string can be sent to a model that will emit a
    single JSON object with yes/no answers for each question.

    :param response: The generated answer that should be examined by the evaluator.
    :param evaluation_questions: A list of dictionaries defining the evaluation
        frameworks and their associated questions.
    :param original_question_prompt: The original user prompt that produced the
        response.

    :return: A multi‑line string comprising the evaluator instructions, the original
        question, the response to assess, and a placeholder where the evaluator
        should insert its JSON output.
    """
    framework_json_parts = _build_json_skeleton(evaluation_questions)

    prompt = f"""### INSTRUCTIONS
        You are an evaluation model. Your task is to evaluate the following response based on the provided evaluation frameworks and their questions.
        For each question, you must answer with "yes" or "no".
        You MUST respond with a single, valid JSON object and nothing else. Do not use markdown formatting.
        
        The JSON object should have the following format:
        {{
        {',\n'.join(framework_json_parts)}
        }}
        
        ### ORIGINAL QUESTION
        {original_question_prompt}
        
        ### RESPONSE TO EVALUATE
        {response}
        
        ### YOUR JSON RESPONSE
        """
    return prompt
