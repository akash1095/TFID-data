from typing import Optional, List, Callable
from enum import Enum
from dataclasses import dataclass, field
import os
import json

from langchain_core.exceptions import OutputParserException
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from pydantic import SecretStr, ValidationError


class GroqModel(Enum):
    """Available Groq models."""

    LLAMA_70B = "llama-3.3-70b-versatile"
    LLAMA_8B = "llama-3.1-8b-instant"
    MIXTRAL = "mixtral-8x7b-32768"
    GEMMA_9B = "gemma2-9b-it"
    DEEPSEEK_R1 = "deepseek-r1-distill-llama-70b"
    GPT_OSS_20B = "openai/gpt-oss-20b"


@dataclass
class LLMConfig:
    """Configuration for LLM inference."""

    model: GroqModel = GroqModel.LLAMA_70B
    temperature: float = 0.3
    max_tokens: Optional[int] = None


class LLMInference:
    """Efficient Groq LLM inference client with structured output support."""

    def __init__(self, api_key: SecretStr, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._api_key = api_key
        self._llm: Optional[ChatGroq] = None
        self._structured_cache: dict = {}

    def llm(self) -> ChatGroq:
        """Lazily initialized Groq LLM client."""
        if self._llm is None:
            self._llm = ChatGroq(
                model=self.config.model.value,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                api_key=self._api_key,
            )
        return self._llm

    def structured_llm(self, schema: type) -> ChatGroq:
        """Get LLM with structured output for a given Pydantic schema."""
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
        """Send prompt to LLM and return response."""
        messages = self._build_messages(prompt, system_prompt)
        return self.llm().invoke(messages).content

    def structured_invoke(
        self, prompt: str, schema: type, system_prompt: Optional[str] = None
    ):
        """Invoke LLM with structured output matching the given Pydantic schema."""
        messages = self._build_messages(prompt, system_prompt)
        return self.structured_llm(schema).invoke(messages)

    async def ainvoke(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Async invoke."""
        messages = self._build_messages(prompt, system_prompt)
        response = await self.llm().ainvoke(messages)
        return response.content

    async def astructured_invoke(
        self, prompt: str, schema: type, system_prompt: Optional[str] = None
    ):
        """Async structured invoke."""
        messages = self._build_messages(prompt, system_prompt)
        return await self.structured_llm(schema).ainvoke(messages)


def get_llm(
    model: str = "llama-3.3-70b-versatile", temperature: float = 0.7
) -> LLMInference:
    model_enum = next((m for m in GroqModel if m.value == model), GroqModel.LLAMA_70B)
    return LLMInference(LLMConfig(model=model_enum, temperature=temperature))
