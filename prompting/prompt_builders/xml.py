from typing import Any
from .interface import BasePromptBuilder

class XMLPromptBuilder(BasePromptBuilder):
    """
    A prompt builder that uses an XML-based template.
    """

    def build_question_prompt(self, question_parts: dict[str, Any]) -> str:
        """
        Builds a question prompt using an XML template.
        """
        system_instructions = question_parts.get("system_instructions", "")
        main_prompt_content = question_parts.get("prompt", "")
        context = question_parts.get("context", [])
        response_option = question_parts.get("response_options", "")

        context_str = "\n".join(c for c in context if c)

        response_options_block = ""
        if response_option:
            response_options_block = f"""
        <response_options>
        {response_option}
        </response_options>
        """

        prompt_template = """<system_instructions>
        {system_instructions}
        </system_instructions>

        <dilemma_prompt>
        {main_prompt_content}
        </dilemma_prompt>

        <context>
        {context}
        </context>
        {response_options_block}
You should also provide a reason after you response with a dash.

        <formatting_instructions>
        Please respond with only one of the options from the <response_options> section. Do not add any other text, explanation, or punctuation.
        </formatting_instructions>"""

        return prompt_template.format(
            system_instructions=system_instructions,
            main_prompt_content=main_prompt_content,
            context=context_str,
            response_options_block=response_options_block,
        )
