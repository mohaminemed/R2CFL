import argparse
import yaml
import pprint 

from src.experiment.serial_runner import FederatedExperiment
from src.experiment.parallel_runner import ParallelFederatedExperiment

def main():
    """
    The main function to run the federated learning experiment.
    """
    # 1. Set up the argument parser
    parser = argparse.ArgumentParser(description="Run a Federated Learning Experiment")
    parser.add_argument(
        '--config', 
        type=str, 
        required=True, 
        help='Path to the YAML configuration file for the experiment.'
    )
    parser.add_argument('--parallel', action='store_true', help='Run client training in parallel.')
    args = parser.parse_args()

    # 2. Load the configuration from the specified YAML file
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {args.config}")
        return
    
    print("--- Experiment Configuration ---")
    pprint.pprint(config)
    print("---------------------------------")

    # 3. Create an instance of the experiment
    if args.parallel:
        experiment = ParallelFederatedExperiment(config)
    else:
        experiment = FederatedExperiment(config)    
    # 4. Run the experiment!
    experiment.run()

if __name__ == "__main__":
    main()