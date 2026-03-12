import json
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import ValidationError


class OllamaModel(Enum):
    """Available Ollama models (common local models)."""

    LLAMA3_8B = "llama3.1:8b"
    LLAMA3_70B = "llama3.1:70b"
    LLAMA3_3_70B = "llama3.3:70b"
    MISTRAL = "mistral:latest"
    MIXTRAL = "mixtral:latest"
    GEMMA2_9B = "gemma2:9b"
    QWEN2_5_7B = "qwen2.5:7b"
    QWEN2_5_14B = "qwen2.5:14b"
    QWEN2_5_32B = "qwen2.5:32b"
    QWEN2_5_72B = "qwen2.5:72b"
    DEEPSEEK_R1_7B = "deepseek-r1:7b"
    DEEPSEEK_R1_8B = "deepseek-r1:8b"
    DEEPSEEK_R1_14B = "deepseek-r1:14b"
    PHI3 = "phi3:latest"
    CODELLAMA = "codellama:latest"


@dataclass
class OllamaConfig:
    """Configuration for Ollama LLM inference."""

    model: OllamaModel = OllamaModel.LLAMA3_8B
    temperature: float = 0.3
    max_tokens: Optional[int] = None
    base_url: str = "http://localhost:11434"  # Default Ollama URL
    num_ctx: Optional[int] = None  # Context window size
    num_predict: Optional[int] = None  # Max tokens to predict
    top_p: Optional[float] = None
    top_k: Optional[int] = None


class OllamaLLMInference:
    """Efficient Ollama LLM inference client with structured output support for local models."""

    def __init__(self, config: Optional[OllamaConfig] = None):
        """
        Initialize Ollama LLM client.

        Args:
            config: OllamaConfig instance. If None, uses default config.
        """
        self.config = config or OllamaConfig()
        self._llm: Optional[ChatOllama] = None
        self._structured_cache: dict = {}

    def llm(self) -> ChatOllama:
        """Lazily initialized Ollama LLM client."""
        if self._llm is None:
            kwargs = {
                "model": self.config.model.value,
                "temperature": self.config.temperature,
                "base_url": self.config.base_url,
            }

            # Add optional parameters if specified
            if self.config.max_tokens is not None:
                kwargs["num_predict"] = self.config.max_tokens
            if self.config.num_ctx is not None:
                kwargs["num_ctx"] = self.config.num_ctx
            if self.config.top_p is not None:
                kwargs["top_p"] = self.config.top_p
            if self.config.top_k is not None:
                kwargs["top_k"] = self.config.top_k

            self._llm = ChatOllama(**kwargs)
        return self._llm

    def structured_llm(self, schema: type) -> ChatOllama:
        """
        Get LLM with structured output for a given Pydantic schema.

        Args:
            schema: Pydantic model class for structured output

        Returns:
            ChatOllama instance configured for structured output
        """
        schema_name = schema.__name__
        if schema_name not in self._structured_cache:
            self._structured_cache[schema_name] = (
                self.llm()
                .with_structured_output(schema)
                .with_retry(
                    retry_if_exception_type=(
                        OutputParserException,
                        ValidationError,
                        json.JSONDecodeError,
                    ),
                    stop_after_attempt=2,
                    wait_exponential_jitter=True,
                )
            )
        return self._structured_cache[schema_name]

    @staticmethod
    def _build_messages(
        prompt: str, system_prompt: Optional[str] = None
    ) -> List[BaseMessage]:
        """Build message list from prompts."""
        messages: List[BaseMessage] = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        return messages

    def invoke(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Send prompt to LLM and return response.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt

        Returns:
            String response from the model
        """
        messages = self._build_messages(prompt, system_prompt)
        return self.llm().invoke(messages).content

    def structured_invoke(
        self, prompt: str, schema: type, system_prompt: Optional[str] = None
    ):
        """
        Invoke LLM with structured output matching the given Pydantic schema.

        Args:
            prompt: User prompt
            schema: Pydantic model class for structured output
            system_prompt: Optional system prompt

        Returns:
            Instance of the schema class with extracted data
        """
        messages = self._build_messages(prompt, system_prompt)
        return self.structured_llm(schema).invoke(messages)

    async def ainvoke(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Async invoke.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt

        Returns:
            String response from the model
        """
        messages = self._build_messages(prompt, system_prompt)
        response = await self.llm().ainvoke(messages)
        return response.content

    async def astructured_invoke(
        self, prompt: str, schema: type, system_prompt: Optional[str] = None
    ):
        """
        Async structured invoke.

        Args:
            prompt: User prompt
            schema: Pydantic model class for structured output
            system_prompt: Optional system prompt

        Returns:
            Instance of the schema class with extracted data
        """
        messages = self._build_messages(prompt, system_prompt)
        return await self.structured_llm(schema).ainvoke(messages)

    def set_model(self, model: OllamaModel):
        """
        Change the model and reset the LLM client.

        Args:
            model: OllamaModel enum value

        Returns:
            self for method chaining
        """
        self.config.model = model
        self._llm = None  # Reset to force re-initialization
        self._structured_cache.clear()
        return self

    def set_temperature(self, temperature: float):
        """
        Change the temperature and reset the LLM client.

        Args:
            temperature: Temperature value (0.0 to 1.0)

        Returns:
            self for method chaining
        """
        self.config.temperature = temperature
        self._llm = None
        self._structured_cache.clear()
        return self


def get_ollama_llm(
    model: str = "llama3.1:8b",
    temperature: float = 0.3,
    base_url: str = "http://localhost:11434",
) -> OllamaLLMInference:
    """
    Factory function to create an OllamaLLMInference instance.

    Args:
        model: Model name (string)
        temperature: Temperature for generation
        base_url: Ollama server URL

    Returns:
        OllamaLLMInference instance
    """
    # Try to match string to enum, fallback to LLAMA3_8B
    model_enum = next(
        (m for m in OllamaModel if m.value == model), OllamaModel.LLAMA3_8B
    )

    config = OllamaConfig(model=model_enum, temperature=temperature, base_url=base_url)

    return OllamaLLMInference(config=config)
