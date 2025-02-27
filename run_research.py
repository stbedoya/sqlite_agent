import logging
from src.utils.config_utils import load_config
from src.execution.research import ExperimentRunner

CONFIG = load_config()
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    experiment = ExperimentRunner(
        model_name=CONFIG["model"]["name"],
        dataset_path=CONFIG["gold_test_set"]["path"],
        output_file=CONFIG["generated_output"]["path"],
        max_examples=2,
        batch_size=2,
    )
    experiment.run()
