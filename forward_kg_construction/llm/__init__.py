# LLM module for forward citation relation extraction
from .llm_inference import get_llm, GroqModel, LLMConfig, LLMInference
from .prompts import (
    EXTRACT_PROMPT_NEW,
    LLAMA_8B_EXTRACT_PROMPT,
    LLAMA_8B_SYSTEM_PROMPT,
)

__all__ = [
    "LLMInference",
    "LLMConfig",
    "GroqModel",
    "get_llm",
    # Prompts
    "EXTRACT_PROMPT_NEW",
    # Llama 8B prompts
    "LLAMA_8B_SYSTEM_PROMPT",
    "LLAMA_8B_EXTRACT_PROMPT",
]
