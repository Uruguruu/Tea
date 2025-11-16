from typing import Any
from .interface import BasePromptBuilder

class DefaultPromptBuilder(BasePromptBuilder):
    """
    The default prompt builder, using a markdown-based template.
    """

    def build_question_prompt(self, question_parts: dict[str, Any]) -> str:
        """
        Builds a question prompt using a markdown template.
        """
        system_instructions = question_parts.get("system_instructions", "")
        prompt = question_parts.get("prompt", "")
        context = question_parts.get("context", [])
        response_options = question_parts.get("response_options", "")

        prompt_parts = []
        if system_instructions:
            prompt_parts.append("### System Instructions\\n" + system_instructions)
        
        prompt_parts.append("### Task\\n" + prompt)

        if context:
            context = [c for c in context if c]
            if context:
                prompt_parts.append("### Context\\n" + "\\n".join(context))

        if response_options:
            prompt_parts.append("### Response Options\\n" + response_options)

        return "\\n\\n".join(prompt_parts)
