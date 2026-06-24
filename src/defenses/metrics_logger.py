import numpy as np
import os
import csv
from typing import List, Set, Optional, Dict

class DefenseMetricsLogger:
    """Handles logging defense metrics (TPR/FPR, etc.) to a CSV file."""
    def __init__(self, output_dir: str, experiment_name: str, defense_name: str):
        os.makedirs(output_dir, exist_ok=True)
        # Create a unique filename based on the experiment and defense name
        filename = f"{experiment_name}_TPR_FPR.csv"
        self.file_path = os.path.join(output_dir, filename)
        
        self.file = open(self.file_path, 'w', newline='')
        self.writer = csv.writer(self.file)
        
        self.headers = [
            'round', 'TP', 'FN', 'FP', 'TN', 
            'TPR_round', 'FPR_round'
        ]
        self.writer.writerow(self.headers)
        print(f"Defense metrics being logged to: {self.file_path}")

    def log_metrics(self, data: Dict):
        """Writes a row of metric data."""
        # Map dictionary keys to CSV columns based on headers
        row = [data.get(h, 'NaN') for h in self.headers]
        self.writer.writerow(row)
        self.file.flush() # Ensure data is written immediately

    def close(self):
        """Closes the underlying file handle."""
        if self.file:
            self.file.close()

class DefenseMetrics:
    """
    Cumulative TPR/FPR tracking and logging capability to defense servers.
    """
    def __init__(self, *args, **kwargs):
        # Call the next class's __init__ in the MRO (e.g., FedAvgAggregator)
        super().__init__(*args, **kwargs) 
        
        # Cumulative Counters
        self.total_tp: int = 0  
        self.total_fn: int = 0 
        self.total_fp: int = 0  
        self.total_tn: int = 0  
        
        self.malicious_ids_known: Set[int] = set() 
        
        # --- Logger Setup ---
        output_dir = kwargs.get('output_dir', 'results')
        experiment_name = kwargs.get('experiment_name', 'default_experiment')
        # Infer defense name from the class name (e.g., MKrumServer -> mkrum)
        defense_name = self.__class__.__name__.lower().replace('server', '') 

        try:
             self.defense_logger = DefenseMetricsLogger(output_dir, experiment_name, defense_name)
        except Exception as e:
             self.defense_logger = None
             print(f"Warning: Could not initialize DefenseMetricsLogger: {e}")

        if not hasattr(self, 'round_counter'):
             self.round_counter = 0


    def set_malicious_ids(self, malicious_ids: List[int]):
         """ Sets the ground truth malicious client IDs for the current round. """
         self.malicious_ids_known = set(malicious_ids)

    def update_defense_metrics(self, 
                               client_ids_received: Set[int], 
                               rejected_client_ids: Set[int]):
        """
        Calculates and updates the cumulative TP, FN, FP, TN counters for the current round,
        and logs the per-round results.
        """
        current_round = self.round_counter + 1
        malicious_ids = self.malicious_ids_known
        
        accepted_client_ids = client_ids_received - rejected_client_ids
        benign_ids = client_ids_received - malicious_ids
        
        # Classification for the current round
        tp_current = len(malicious_ids.intersection(rejected_client_ids))      # True Positive
        fn_current = len(malicious_ids.intersection(accepted_client_ids))       # False Negative
        fp_current = len(benign_ids.intersection(rejected_client_ids))         # False Positive
        tn_current = len(benign_ids.intersection(accepted_client_ids))          # True Negative

        # Update cumulative totals
        self.total_tp += tp_current
        self.total_fn += fn_current
        self.total_fp += fp_current
        self.total_tn += tn_current

        # --- Round-by-Round Reporting & Logging ---
        total_mal_round = tp_current + fn_current
        total_ben_round = fp_current + tn_current
        
        round_tpr = tp_current / total_mal_round if total_mal_round > 0 else np.nan
        round_fpr = fp_current / total_ben_round if total_ben_round > 0 else np.nan

        log_data = {
            'round': current_round, 
            'TP': tp_current, 'FN': fn_current, 'FP': fp_current, 'TN': tn_current,
            'TPR_round': round_tpr, 'FPR_round': round_fpr
        }
        
        if self.defense_logger:
            self.defense_logger.log_metrics(log_data)
            
        print(f"--- Round {current_round} Defense Metrics ---")
        print(f"  TP={tp_current}, FN={fn_current}, FP={fp_current}, TN={tn_current}")
        print(f"  TPR (Round): {round_tpr:.4f}, FPR (Round): {round_fpr:.4f}")
        

    def close(self):
         """ Overload the close method for final reporting and file cleanup. """
         self.report_final_metrics()
         # Call base class close if it exists
         if hasattr(super(), 'close'):
             super().close()
         # Close the defense logger file
         if self.defense_logger:
             self.defense_logger.close()
             print(f"Closed defense metrics logger: {self.defense_logger.file_path}")

    def report_final_metrics(self):
        """ Computes and prints the final cumulative metrics for the entire experiment. """
        total_malicious = self.total_tp + self.total_fn
        total_benign = self.total_fp + self.total_tn

        final_tpr = self.total_tp / total_malicious if total_malicious > 0 else 0.0
        final_fpr = self.total_fp / total_benign if total_benign > 0 else 0.0

        # --- LOG FINAL CUMULATIVE METRICS TO FILE ---
        log_data_final = {
            'round': "FINAL_CUMULATIVE", 
            'TP': self.total_tp, 'FN': self.total_fn, 'FP': self.total_fp, 'TN': self.total_tn,
            'TPR_round': final_tpr, 'FPR_round': final_fpr 
        }
        if self.defense_logger:
            self.defense_logger.log_metrics(log_data_final)
        # --------------------------------------------

        print(f"\n======================================")
        print(f" Defense Final Cumulative Metrics")
        print(f"======================================")
        print(f" Final True Positive Rate (TPR): {final_tpr:.4f}")
        print(f" Final False Positive Rate (FPR): {final_fpr:.4f}")
        print(f"======================================")
        return final_tpr, final_fpr