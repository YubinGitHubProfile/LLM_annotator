# a helper class to help annotating batches of datapoints in a csv file using the llm agents.
import csv
import json

class CSVparser:
    def __init__(self, csv_file, batch_size=5, llm_agents=None):
        """
        Initializes the CSVLoader.

        Args:
          csv_file: Path to the CSV file.
          batch_size: Number of data points to process per batch.
          llm_api_function: The function to call for LLM annotation. 
                            This function should take a string as input and return a JSON string.
        """
        self.csv_file = csv_file
        self.batch_size = batch_size
        self.llm_api_function = llm_agents
        if not self.llm_api_function:
            raise ValueError("You must provide an llm_agents function which takes input of a batch of texts and output a json or a list of dictionaries.")

    def load_batch(self, reader):
        """
        Loads a batch of data points from the CSV file.

        Args:
          reader: A CSV reader object.

        Returns:
          A list of dictionaries, where each dictionary represents a data point.
        """
        batch = []
        try:
            for _ in range(self.batch_size):
                row = next(reader)
                # Create a dictionary using the header from the annotate_csv method
                batch.append(dict(zip(self.header, row)))  # Use self.header instead of re-reading it
        except StopIteration:
            pass  # Reached the end of the file
        return batch

    def prepare_llm_input(self, batch, text_column="UserInput"): # text_column can be changed
        """
        Formats a batch of strings into a JSON-like structure for LLM input.

        Args:
          batch: A list of dictionaries representing a batch of data points.
          text_column: The column containing the text data (default is 'UserInput').

        Returns:
          A list of dictionaries in the format [{"text": "string1"}, {"text": "string2"}].
        """
        # Validate and format the batch into the desired structure
        json_batch = []
        for data_point in batch:
            if text_column in data_point:  # Validate column presence
                json_batch.append({"text": data_point[text_column]})
            else:
                raise ValueError(f"Column '{text_column}' not found in the data point: {data_point}")
        return json_batch

    def annotate_batch(self, batch):
        """
        Annotates a batch of data points using the defined LLM agents.

        Args:
          batch: A list of dictionaries representing a batch of data points.

        Returns:
          A list of JSON strings, where each string contains the annotations for a data point.
        """
        llm_input = self.prepare_llm_input(batch, text_column="UserInput")
        llm_output = self.llm_api_function.annotate(batch_message=llm_input)
        # Assuming the LLM API returns a list of JSON strings
        # Ensure the output is a list of dictionaries
        if isinstance(llm_output, str):
            try:
                llm_output = json.loads(llm_output)
                if not isinstance(llm_output, list):
                    raise ValueError("LLM output must be a list of dictionaries.")
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to decode LLM output: {e}")
        return llm_output

    def write_annotations(self, annotations, output_file):
        """
        Writes the annotations to a new CSV file, even with unknown annotation keys.

        Args:
          annotations: A list of dictionaries, where each dictionary contains annotations.
          output_file: Path to the output CSV file.
        """
        with open(self.csv_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
            reader = csv.reader(infile)
            writer = csv.writer(outfile)

            # Write header row (assuming original CSV has a header)
            header = next(reader)

             # Reconcile all keys across annotations for consistent column output
            all_keys = set()
            for annotation in annotations:
                all_keys.update(annotation.keys())
            all_keys = sorted(all_keys)  # Optional: Sort keys for consistent ordering

            header.extend(all_keys)
            writer.writerow(header)

            annotation_count = 0
            for row in reader:
                if annotation_count < len(annotations):
                    annotation_data = annotations[annotation_count]
                    # Extract annotation values in the order of the keys
                    row.extend([annotation_data.get(key, "") for key in all_keys]) 
                else:
                    # If no more annotations, add empty values for the annotation columns
                    row.extend([""] * len(all_keys)) 

                writer.writerow(row)
                annotation_count += 1

    def annotate_csv(self, output_file):
        """
        Annotates the entire CSV file.

        Args:
          output_file: Path to the output CSV file.
        """
        all_annotations = []
        with open(self.csv_file, 'r') as infile:
            reader = csv.reader(infile)
            self.header = next(reader)  # Skip header row
            print("CSV Header:", self.header) # debugging line
            
            i = 1
            while True:
                batch = self.load_batch(reader)
                print(f"Batch {i} data loaded.")
                if not batch:
                    print("Annotation completed. Batch data loading ended.")
                    break
                # Annotate and handle errors
                try:
                    annotations = self.annotate_batch(batch)  # Use batch_dicts for annotation
                    print(f"Batch {i} data annotated!") # debugging line
                    all_annotations.extend(annotations)  # Extend the list instead of appending
                except Exception as e:
                    print(f"Error processing batch: {e}")
                i+=1
        self.write_annotations(all_annotations, output_file)

######see the colab notebook for example usage#####

