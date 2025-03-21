import logging
import jsonlines

from src.llm.llm_model import LLMModel
from src.llm.llm_prompt import PromptLoader
from src.llm.sql_extractor import SQLQueryExtractor
from src.research.dataset_loader import GoldDatasetLoader
from src.utils.schema_provider import SchemaProvider

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Run pipeline with gold dataset."""

    def __init__(
        self,
        model_name: str,
        dataset_path: str,
        output_file: str,
        prompt_file: str,
        experiment_name: str,
        max_examples: int = None,
        batch_size: int = 100,
    ):
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.output_file = output_file
        self.max_examples = max_examples
        self.batch_size = batch_size
        self.experiment_name = experiment_name
        self.llm_model = LLMModel(model_name)
        self.prompt_loader = PromptLoader(prompt_file)

    def run(self):
        """Run the experiment and save results in batches to optimize performance."""
        batch = []

        with jsonlines.open(self.output_file, mode="a") as writer:
            for example in GoldDatasetLoader.load_gold_dataset(
                self.dataset_path, self.max_examples
            ):
                question = example.get("question", "")
                variables = {
                    "schema": SchemaProvider.get_nba_schema(),
                    "question": question,
                }

                prompt = self.prompt_loader.get_prompt(self.experiment_name, variables)
                logger.info(f"Processing question: {question}")

                output_llm = self.llm_model.generate_response(prompt)
                logger.info(f"Generated Response: {output_llm}")

                sql_query = SQLQueryExtractor.extract_sql_query(output_llm)
                logger.info(f"Extracted SQL Query: {sql_query}")

                # tokens = tokenizer.tokenize(text)
                # num_tokens = len(tokens)
                # print(f"Number of tokens in your text: {num_tokens}")

                batch.append(
                    {
                        "question": question,
                        "generated_response": output_llm,
                        "sql_query": sql_query,
                        "num_output_tokens": len(output_llm.split()),
                        "num_input_tokens": len(prompt.split()),
                    }
                )

                if len(batch) >= self.batch_size:
                    writer.write_all(batch)
                    batch.clear()
                    logger.info(
                        f"Saved {self.batch_size} results to {self.output_file}"
                    )

            if batch:
                writer.write_all(batch)
                logger.info(f"Saved final {len(batch)} results to {self.output_file}")
        self.llm_model.close()
