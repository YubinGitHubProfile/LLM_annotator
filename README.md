# An Annotation Tool with Gemini LLM

This tool allows you to annotate texts or texts in CSV files using the power of Gemini LLM and a chain of prompts. It's designed to be flexible and can be customized for various annotation tasks.

## Modules

* **Prompt:** Represents a single prompt with placeholders for dynamic data.
* **PromptChain:** Manages a sequence of Prompts to guide the annotation process.
* **Annotator:** Handles the interaction with the Gemini LLM API. This can be expanded into several other agents based on your needs.
* **CSVparser:** Loads and processes CSV files, feeding data to the Annotator in batches.

## Example Usage (Google Colab)

```python
# Import necessary modules
from agents.CreatePrompt import Prompt, PromptChain
from agents.Annotator import Annotator
from data.data_loader import CSVparser
import os
import google.generativeai as genai
from google.colab import userdata

# Set up paths and API key (replace with your actual paths)
base_dir = '/content/drive/MyDrive/annotators' 
input_csv = os.path.join(base_dir, "task1/input.csv") 
output_csv = os.path.join(base_dir, "task1/output_annotated.csv")
prompt1_path = os.path.join(base_dir, "prompts/task1/prompt1.txt")
prompt2_path = os.path.join(base_dir, "prompts/task1/prompt2.txt")
prompt3_path = os.path.join(base_dir, "prompts/task1/prompt3.txt")
api_key = userdata.get('GOOGLE_API_KEY')
model = 'gemini-2.0-flash-exp' 

# Load prompts from files
with open(prompt1_path, 'r') as file:
    prompt1_text = file.read()
with open(prompt2_path, 'r') as file:
    prompt2_text = file.read()
with open(prompt3_path, 'r') as file:
    prompt3_text = file.read()

# Create Prompt objects
prompt1 = Prompt(prompt1_text)
prompt2 = Prompt(prompt2_text)
prompt3 = Prompt(prompt3_text)

# Build the PromptChain
chain = PromptChain()
chain.add_prompt(prompt1)
chain.add_prompt(prompt2)
chain.add_prompt(prompt3)

# Initialize the Annotator
annotator = Annotator(prompt_chain=chain, api_key=api_key, gemini_model=model)

# Initialize the CSVparser
batch_size = 5  
csv_parser = CSVparser(csv_file=input_csv, batch_size=batch_size, llm_agents=annotator)

# Annotate the CSV file
csv_parser.annotate_csv(output_file=output_csv)
```
## Explanation:
Import Modules: Import the necessary modules for prompts, annotation, and CSV handling.
Set Up Paths and API Key: Define the paths to your input CSV, output CSV, and prompt files. Also, provide your Gemini API key.
Load Prompts: Load the raw prompt text from the specified files.
Create Prompt Objects: Create Prompt objects for each prompt in your annotation chain.
Build PromptChain: Create a PromptChain object and add your Prompt objects to it in the desired order.
Initialize Annotator: Create an Annotator object, providing the PromptChain, API key, and the desired Gemini model.
Initialize CSVparser: Create a CSVparser object, specifying the input CSV file, batch size for processing, and the Annotator object.
Annotate CSV: Call the annotate_csv method of the CSVparser to process the input CSV file and save the annotated output to the specified output file.

## Key Considerations:
Prompt Design: Carefully design your prompts to guide the LLM towards the desired annotations. Use clear instructions and examples.
Batch Size: Adjust the batch_size parameter in CSVparser based on the complexity of your prompts and the capabilities of your system.
Error Handling: Implement error handling to gracefully manage potential issues during API calls or file processing.
Customization: The tool can be easily adapted to different annotation tasks by modifying the prompts and the PromptChain.
