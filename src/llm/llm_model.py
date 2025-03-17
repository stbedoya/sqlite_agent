import gc
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils.config_utils import load_config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CONFIG = load_config()


class LLMModel:
    """Class to handle model loading and inference."""

    def __init__(self, model_name: str):
        model_path = CONFIG["model"]["path"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.release_memory()

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        logger.info(f"Model loaded from: {self.model.config._name_or_path}")
        logger.info(
            f"Model successfully loaded on device: {self.model.device}"
        )

    def generate_response(self, prompt: str, max_tokens: int = None) -> str:
        """Generate response from LLM, excluding system prompt and user input."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(
            self.model.device
        )

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                # max_new_tokens=max_tokens,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.2,
                # top_p=0.9,
            )

        response_tokens = output[0][inputs.input_ids.shape[-1] :]
        return self.tokenizer.decode(response_tokens, skip_special_tokens=True)

    def release_memory(self):
        """Safely release GPU memory."""
        logger.info("Releasing GPU memory...")
        if "torch" in globals() and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def close(self):
        """Explicit method to release resources before deleting the object."""
        self.release_memory()
        logger.info("LLMModel instance closed.")

    def __del__(self):
        """Ensure safe cleanup during garbage collection."""
        try:
            self.release_memory()
            logger.info("LLMModel instance deleted and memory released.")
        except Exception as e:
            logger.warning(f"Exception during LLMModel cleanup: {e}")
