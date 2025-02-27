import gc
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class LLMModel:
    """Class to handle model loading and inference."""

    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.release_memory()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        logger.info(f"Model loaded successfully on device: {self.model.device}")

    def generate_response(self, prompt: str, max_tokens: int = 200) -> str:
        """Generate response from LLM."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def release_memory(self):
        logger.info("Releasing GPU memory...")
        if hasattr(self, "model") and self.model is not None:
            torch.cuda.empty_cache()
        gc.collect()

    def __del__(self):
        self.release_memory()
        logger.info("LLMModel instance deleted and memory released.")
