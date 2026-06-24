# In src/experiments/logging.py

import csv
from typing import Dict, List, Optional
import os

class MetricsLogger:
    """
    A simple logger to save experiment metrics to a CSV file.
    """
    def __init__(self, output_dir: str, experiment_name: str, headers: List[str]):
        """
        Initializes the logger and creates the CSV file with headers.

        Args:
            output_dir (str): The directory to save the log file in.
            experiment_name (str): The name of the experiment, used for the filename.
            headers (List[str]): A list of column headers for the CSV file.
        """
        self.headers = headers
        
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"{experiment_name}_metrics.csv")
        
        # Open the file in write mode. 'newline=""' is important for csv writer.
        self.file = open(filepath, 'w', newline='')
        self.writer = csv.writer(self.file)
        
        # Write the header row immediately
        self.writer.writerow(self.headers)

    def log_round(self, round_data: Dict):
        """
        Writes a single row of data for a completed round.

        Args:
            round_data (Dict): A dictionary of metrics for the round. Keys
                should match the headers provided during initialization.
        """
        # Create a row by looking up each header in the data dictionary
        row = [round_data.get(h, '') for h in self.headers]
        self.writer.writerow(row)
        # Flush the buffer to ensure data is written to disk immediately
        self.file.flush()

    def close(self):
        """Closes the file handle."""
        self.file.close()