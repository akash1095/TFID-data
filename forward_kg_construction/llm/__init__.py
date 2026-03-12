# LLM module for forward citation relation extraction
from .llm_inference import get_llm, GroqModel, LLMConfig, LLMInference
from .openai_inference import (
    OpenAIInference,
    OpenAIConfig,
    OpenAIModel,
    get_vllm_client,
    get_openai_client,
)
from .prompts import (
    EXTRACT_PROMPT_NEW,
    LLAMA_8B_EXTRACT_PROMPT,
    LLAMA_8B_SYSTEM_PROMPT,
)

__all__ = [
    # Groq
    "LLMInference",
    "LLMConfig",
    "GroqModel",
    "get_llm",
    # OpenAI/vLLM
    "OpenAIInference",
    "OpenAIConfig",
    "OpenAIModel",
    "get_vllm_client",
    "get_openai_client",
    # Prompts
    "EXTRACT_PROMPT_NEW",
    "LLAMA_8B_SYSTEM_PROMPT",
    "LLAMA_8B_EXTRACT_PROMPT",
]
