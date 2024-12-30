class Prompt:
    def __init__(self, text, role="user"):
        """
        Initialize a Prompt with text and a role.
        Args:
            text (str): The text of the prompt.
            role (str): The role associated with the prompt (default is "user").
        """
        self.text = text
        self.role = role

    def format(self, **kwargs):
        """
        Format the prompt with provided keyword arguments.
        Args:
            kwargs: Formatting parameters for the text.
        Returns:
            str: Formatted text.
        """
        return self.text.format(**kwargs)


class PromptChain:
    def __init__(self, prompts=None):
        """
        Initialize a PromptChain with a list of prompts.
        Args:
            prompts (list): List of Prompt objects (default is an empty list).
        """
        self.prompts = prompts if prompts else []

    def add_prompt(self, prompt):
        """
        Add a Prompt object to the chain.
        Args:
            prompt (Prompt): The Prompt object to add.
        """
        self.prompts.append(prompt)

    def format_prompts(self, **kwargs):
        """
        Format all prompts in the chain with the provided arguments.
        Args:
            kwargs: Formatting parameters for each prompt in the chain.
        Returns:
            list: A list of formatted prompt texts.
        """
        return [prompt.format(**kwargs) for prompt in self.prompts]