# Annotate a batch of texts

import json
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

class Annotator:
    def __init__(self, prompt_chain, api_key, gemini_model):
        """
        Initialize the Annotator with a PromptChain, API key, and model.
        Args:
            prompt_chain (PromptChain): A chain of prompts created using the PromptChain class.
            api_key (str): API key for Google Generative AI.
            gemini_model (str): Name of the Gemini model to use.
        """
        self.prompt_chain = prompt_chain
        self.api_key = api_key
        self.model = gemini_model
    
    def run_chain(self, context=None, batch_message=None, **kwargs):
        """
        Generate annotations by running the chain of prompts.
        Args:
            context (str): Initial input to pass to the chain (optional).
            kwargs: Formatting parameters for the prompts.
        Returns:
            str: Final response after processing all prompts in the chain.
        """

        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model)
        response = context or ""
        # Ensure batch_message is passed in kwargs to be used in the first prompt
        kwargs['batch_message'] = batch_message
        # i=0 # debugging line
        for formatted_prompt in self.prompt_chain.format_prompts(**kwargs):
            full_prompt = f"{response}\n\n{formatted_prompt}"
            # print(f"Full prompt {i+1} is:\n\n", full_prompt) # debugging line
            response = model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    candidate_count=1,
                    max_output_tokens=5000,
                    temperature=0,
                    # top_k=100,
                    # top_p=0.95,
                ),
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    # HarmCategory.HARM_CATEGORY_VIOLENCE: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    # HarmCategory.HARM_CATEGORY_TOXICITY: HarmBlockThreshold.BLOCK_NONE,
                }
              ).text
            # print(f"Full response {i+1} is:\n\n", response) # debugging line
            # i+=1 # debugging line
        return response

    def parse_json(self, text):
        """
        Parse a JSON string from the model's output.
        Args:
            text (str): The text output from the model.
        Returns:
            dict or list: Parsed JSON data, or an empty list if parsing fails.
        """
        json_string = text
        if json_string.startswith("```json"):
            json_string = json_string[len("```json"):].strip()
        if json_string.endswith("```"):
            json_string = json_string[:-len("```")].strip()

        try:
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return []

    def annotate(self, context=None, **kwargs):
        """
        Generate annotations and return the parsed JSON result.
        Args:
            context (str): Initial input to pass to the chain (optional).
            kwargs: Formatting parameters for the prompts.
        Returns:
            dict or list: Parsed JSON data from the chain's output.
        """
        output = self.run_chain(context, **kwargs)
        return self.parse_json(output)