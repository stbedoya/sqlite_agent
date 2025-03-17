from langchain_core.prompts import PromptTemplate
from typing import Dict, Any


class PromptBuilder:
    def __init__(self, system_instructions: str, prompt: str):
        """
        Initialize the PromptBuilder with system instructions and the main prompt.
        Utilizes PromptTemplate for handling prompt variable substitution.
        """
        self.prompt_template = PromptTemplate.from_template(prompt)
        self.system_instructions = system_instructions
        self.variables = {}

    def add_variables(self, variables: Dict[str, Any]):
        """
        Add or update multiple input variables (for both system instructions and the prompt).
        """
        self.variables.update(variables)
        self.prompt_template.input_variables = list(self.variables.keys())

    def build_prompt(self) -> str:
        """
        Replace placeholders in the system instructions and the prompt with actual values from the input variables.
        The system instructions are treated as a variable inside the prompt.
        """
        formatted_system_instructions = self.system_instructions.format(
            **self.variables
        )
        self.variables["system_instructions"] = formatted_system_instructions
        formatted_prompt = self.prompt_template.format(**self.variables)

        return formatted_prompt
