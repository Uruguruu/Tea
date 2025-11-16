import abc
import textwrap
from abc import ABC
from typing import Any, List


class PromptBuilderInterface(metaclass=abc.ABCMeta):
    """
    A formal interface for any class that can build a prompt.
    """

    @abc.abstractmethod
    def build_question_prompt(self, question_parts: dict[str, Any]) -> str:
        """
        Builds a question prompt from a dictionary of parts.
        """
        pass

    @abc.abstractmethod
    def build_evaluation_prompt(
        self,
        response: str,
        evaluation_questions: List[dict[str, Any]],
        original_question_prompt: str,
    ) -> str:
        """
        Builds an evaluation prompt.
        """
        pass


class BasePromptBuilder(PromptBuilderInterface, ABC):
    """
    A base class for prompt builders that provides a default implementation
    for the evaluation prompt.
    """

    def build_evaluation_prompt(
        self,
        response: str,
        evaluation_questions: List[dict[str, Any]],
        original_question_prompt: str,
    ) -> str:
        """
        Builds an evaluation prompt.
        """
        framework_json_parts = self._build_json_skeleton(evaluation_questions)

        prompt = textwrap.dedent(f"""\
            ### INSTRUCTIONS
            You are an evaluation model. Your task is to evaluate the following response based on the provided evaluation frameworks and their questions.
            For each question, you must answer with 'yes' or 'no'.
            You MUST respond with a single, valid JSON object and nothing else. Do not use markdown formatting.
            
            The JSON object should have the following format:
            {{
            {',\\n'.join(framework_json_parts)}
            }}
            
            ### ORIGINAL QUESTION
            {original_question_prompt}
            
            ### RESPONSE TO EVALUATE
            {response}
            
            ### YOUR JSON RESPONSE
            """)

        return prompt

    @staticmethod
    def _build_json_skeleton(evaluation_questions: list[dict[str, Any]]) -> list[str]:
        """
        Build a JSON‑compatible skeleton from a collection of evaluation questions.
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
                f'  "{framework_name}": {{\\n' + ",\\n".join(question_lines) + "\\n  }"
            )
        return framework_json_parts
