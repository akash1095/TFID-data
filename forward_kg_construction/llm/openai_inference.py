"""
OpenAI-compatible LLM inference client for vLLM and OpenAI API.

Supports:
- invoke/ainvoke for single requests
- batch/abatch for batch requests (FAST!)
- Structured output with Pydantic
- Automatic retries
- vLLM-specific optimizations (repetition_penalty, guided_json)
"""

import json
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Any

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import ValidationError


class OpenAIModel(Enum):
    """Available OpenAI-compatible models."""
    
    # OpenAI models
    GPT4_TURBO = "gpt-4-turbo-preview"
    GPT4 = "gpt-4"
    GPT35_TURBO = "gpt-3.5-turbo"
    
    # vLLM models (HuggingFace format)
    QWEN_2_5_7B = "Qwen/Qwen2.5-7B-Instruct"
    QWEN_2_5_14B = "Qwen/Qwen2.5-14B-Instruct"
    QWEN_2_5_32B = "Qwen/Qwen2.5-32B-Instruct"
    QWEN_2_5_72B = "Qwen/Qwen2.5-72B-Instruct"
    LLAMA_3_1_8B = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    LLAMA_3_1_70B = "meta-llama/Meta-Llama-3.1-70B-Instruct"
    MISTRAL_7B = "mistralai/Mistral-7B-Instruct-v0.3"
    
    # vLLM served models (custom names)
    VLLM_QWEN_7B = "qwen2.5-7b"
    VLLM_QWEN_14B = "qwen2.5-14b"
    VLLM_LLAMA_8B = "llama3.1-8b"


@dataclass
class OpenAIConfig:
    """Configuration for OpenAI-compatible LLM inference."""
    
    model: str = "qwen2.5-7b"
    temperature: float = 0.3
    max_tokens: int = 512
    base_url: str = "http://localhost:8000/v1"  # vLLM default
    api_key: str = "EMPTY"  # vLLM doesn't need key
    
    # Sampling parameters
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 40
    
    # Anti-repetition (vLLM specific)
    repetition_penalty: Optional[float] = 1.1
    frequency_penalty: Optional[float] = 0.2
    presence_penalty: Optional[float] = 0.1
    
    # Performance
    timeout: int = 120
    max_retries: int = 3
    
    # vLLM specific
    use_guided_json: bool = False  # Force valid JSON output


class OpenAIInference:
    """
    OpenAI-compatible LLM inference client with batch support.
    
    Works with:
    - vLLM (local or remote)
    - OpenAI API
    - Any OpenAI-compatible endpoint
    
    Features:
    - invoke/ainvoke for single requests
    - batch/abatch for batch requests (3-5x faster!)
    - Structured output with Pydantic
    - Automatic retries
    """
    
    def __init__(self, config: Optional[OpenAIConfig] = None):
        """
        Initialize OpenAI-compatible LLM client.
        
        Args:
            config: OpenAIConfig instance. If None, uses default config.
        """
        self.config = config or OpenAIConfig()
        self._llm: Optional[ChatOpenAI] = None
        self._structured_cache: Dict[str, ChatOpenAI] = {}
    
    def llm(self) -> ChatOpenAI:
        """Lazily initialized ChatOpenAI client."""
        if self._llm is None:
            # Base parameters
            kwargs = {
                "model": self.config.model,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "base_url": self.config.base_url,
                "api_key": self.config.api_key,
                "timeout": self.config.timeout,
                "max_retries": self.config.max_retries,
            }
            
            # Model-specific parameters (for vLLM)
            model_kwargs = {}
            
            if self.config.top_p is not None:
                model_kwargs["top_p"] = self.config.top_p
            if self.config.top_k is not None:
                model_kwargs["top_k"] = self.config.top_k
            if self.config.repetition_penalty is not None:
                model_kwargs["repetition_penalty"] = self.config.repetition_penalty
            if self.config.frequency_penalty is not None:
                kwargs["frequency_penalty"] = self.config.frequency_penalty
            if self.config.presence_penalty is not None:
                kwargs["presence_penalty"] = self.config.presence_penalty
            
            if model_kwargs:
                kwargs["model_kwargs"] = model_kwargs
            
            self._llm = ChatOpenAI(**kwargs)
        return self._llm
    
    def structured_llm(self, schema: type, use_guided_json: bool = None) -> ChatOpenAI:
        """
        Get LLM with structured output for a given Pydantic schema.
        
        Args:
            schema: Pydantic model class for structured output
            use_guided_json: Override config.use_guided_json
        
        Returns:
            ChatOpenAI instance configured for structured output
        """
        use_guided = use_guided_json if use_guided_json is not None else self.config.use_guided_json
        cache_key = f"{schema.__name__}_guided_{use_guided}"
        
        if cache_key not in self._structured_cache:
            llm = self.llm()

            # Add guided JSON if using vLLM
            if use_guided:
                # Create new LLM instance with guided_json
                kwargs = llm._default_params.copy()
                if "model_kwargs" not in kwargs:
                    kwargs["model_kwargs"] = {}
                kwargs["model_kwargs"]["guided_json"] = schema.schema_json()

                llm = ChatOpenAI(**kwargs)

            # Add structured output and retry logic
            self._structured_cache[cache_key] = (
                llm
                .with_structured_output(schema)
                .with_retry(
                    retry_if_exception_type=(
                        OutputParserException,
                        ValidationError,
                        json.JSONDecodeError,
                    ),
                    stop_after_attempt=self.config.max_retries,
                    wait_exponential_jitter=True,
                )
            )
        return self._structured_cache[cache_key]

    def build_messages(
        self,
        system_prompt: str,
        user_prompt: str,
        additional_messages: Optional[List[BaseMessage]] = None
    ) -> List[BaseMessage]:
        """
        Build message list for LLM.

        Args:
            system_prompt: System instruction
            user_prompt: User query/input
            additional_messages: Optional additional messages

        Returns:
            List of messages ready for LLM
        """
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        if additional_messages:
            messages.extend(additional_messages)

        return messages

    # ========== Sync Methods ==========

    def invoke(
        self,
        messages: List[BaseMessage],
        schema: Optional[type] = None,
        **kwargs
    ) -> Any:
        """
        Invoke LLM with single request (sync).

        Args:
            messages: List of messages
            schema: Optional Pydantic schema for structured output
            **kwargs: Additional parameters

        Returns:
            LLM response (structured if schema provided)
        """
        if schema:
            llm = self.structured_llm(schema)
        else:
            llm = self.llm()

        return llm.invoke(messages, **kwargs)

    def batch(
        self,
        messages_list: List[List[BaseMessage]],
        schema: Optional[type] = None,
        **kwargs
    ) -> List[Any]:
        """
        Batch invoke LLM with multiple requests (sync).

        Args:
            messages_list: List of message lists
            schema: Optional Pydantic schema for structured output
            **kwargs: Additional parameters

        Returns:
            List of LLM responses
        """
        if schema:
            llm = self.structured_llm(schema)
        else:
            llm = self.llm()

        return llm.batch(messages_list, **kwargs)

    # ========== Async Methods (FAST!) ==========

    async def ainvoke(
        self,
        messages: List[BaseMessage],
        schema: Optional[type] = None,
        **kwargs
    ) -> Any:
        """
        Async invoke LLM with single request.

        Args:
            messages: List of messages
            schema: Optional Pydantic schema for structured output
            **kwargs: Additional parameters

        Returns:
            LLM response (structured if schema provided)
        """
        if schema:
            llm = self.structured_llm(schema)
        else:
            llm = self.llm()

        return await llm.ainvoke(messages, **kwargs)

    async def abatch(
        self,
        messages_list: List[List[BaseMessage]],
        schema: Optional[type] = None,
        max_concurrency: int = 16,
        **kwargs
    ) -> List[Any]:
        """
        Async batch invoke LLM with multiple requests (FASTEST!).

        This is the FASTEST method for processing many requests.
        Uses async + batching for maximum throughput.

        Args:
            messages_list: List of message lists
            schema: Optional Pydantic schema for structured output
            max_concurrency: Max concurrent requests (default: 16)
            **kwargs: Additional parameters

        Returns:
            List of LLM responses
        """
        if schema:
            llm = self.structured_llm(schema)
        else:
            llm = self.llm()

        # LangChain's abatch handles concurrency internally
        config = {"max_concurrency": max_concurrency}
        return await llm.abatch(messages_list, config=config, **kwargs)

    # ========== Convenience Methods ==========

    def extract_relationship(
        self,
        system_prompt: str,
        citing_title: str,
        citing_abstract: str,
        cited_title: str,
        cited_abstract: str,
        schema: type
    ) -> Any:
        """
        Extract relationship between two papers (sync).

        Args:
            system_prompt: System instruction
            citing_title: Title of citing paper
            citing_abstract: Abstract of citing paper
            cited_title: Title of cited paper
            cited_abstract: Abstract of cited paper
            schema: Pydantic schema for output

        Returns:
            Structured relationship analysis
        """
        user_prompt = f"""CITING PAPER:
Title: {citing_title}
Abstract: {citing_abstract}

CITED PAPER:
Title: {cited_title}
Abstract: {cited_abstract}"""

        messages = self.build_messages(system_prompt, user_prompt)
        return self.invoke(messages, schema=schema)

    async def extract_relationship_async(
        self,
        system_prompt: str,
        citing_title: str,
        citing_abstract: str,
        cited_title: str,
        cited_abstract: str,
        schema: type
    ) -> Any:
        """
        Extract relationship between two papers (async).

        Args:
            system_prompt: System instruction
            citing_title: Title of citing paper
            citing_abstract: Abstract of citing paper
            cited_title: Title of cited paper
            cited_abstract: Abstract of cited paper
            schema: Pydantic schema for output

        Returns:
            Structured relationship analysis
        """
        user_prompt = f"""CITING PAPER:
Title: {citing_title}
Abstract: {citing_abstract}

CITED PAPER:
Title: {cited_title}
Abstract: {cited_abstract}"""

        messages = self.build_messages(system_prompt, user_prompt)
        return await self.ainvoke(messages, schema=schema)


# ========== Factory Functions ==========

def get_vllm_client(
    model: str = "qwen2.5-7b",
    base_url: str = "http://localhost:8000/v1",
    temperature: float = 0.3,
    max_tokens: int = 512,
    use_guided_json: bool = True,
) -> OpenAIInference:
    """
    Factory function to create vLLM client.

    Args:
        model: Model name (as served by vLLM)
        base_url: vLLM server URL
        temperature: Temperature for generation
        max_tokens: Max tokens to generate
        use_guided_json: Use guided JSON generation (recommended)

    Returns:
        OpenAIInference instance configured for vLLM
    """
    config = OpenAIConfig(
        model=model,
        base_url=base_url,
        api_key="EMPTY",
        temperature=temperature,
        max_tokens=max_tokens,
        repetition_penalty=1.1,  # Fix repeated tokens
        frequency_penalty=0.2,
        presence_penalty=0.1,
        use_guided_json=use_guided_json,
    )

    return OpenAIInference(config=config)


def get_openai_client(
    model: str = "gpt-4-turbo-preview",
    api_key: str = None,
    temperature: float = 0.3,
    max_tokens: int = 512,
) -> OpenAIInference:
    """
    Factory function to create OpenAI API client.

    Args:
        model: OpenAI model name
        api_key: OpenAI API key
        temperature: Temperature for generation
        max_tokens: Max tokens to generate

    Returns:
        OpenAIInference instance configured for OpenAI
    """
    import os

    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")

    config = OpenAIConfig(
        model=model,
        base_url="https://api.openai.com/v1",
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        use_guided_json=False,  # OpenAI doesn't support guided JSON
    )

    return OpenAIInference(config=config)


