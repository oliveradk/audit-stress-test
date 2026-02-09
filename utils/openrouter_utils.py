"""Generic async utilities for OpenRouter API calls.

Provides async API calls through OpenRouter with batching, retry logic,
and client lifecycle management.
"""

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Awaitable, Callable, List, TypeVar

import openai
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Configuration
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Retry configuration
RETRY_ATTEMPTS = 5
RETRY_MIN_WAIT = 1
RETRY_MAX_WAIT = 60

T = TypeVar("T")
R = TypeVar("R")


# =============================================================================
# Client Management
# =============================================================================

def get_client() -> AsyncOpenAI:
    """Get configured AsyncOpenAI client for OpenRouter."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment")
    return AsyncOpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        default_headers={
            "HTTP-Referer": "https://github.com/audit-stress-test",
            "X-Title": "Auditing Stress Tests",
        },
    )


@asynccontextmanager
async def client():
    """Context manager for OpenRouter client with automatic cleanup."""
    c = get_client()
    try:
        yield c
    finally:
        await c.close()


# =============================================================================
# Core Completion
# =============================================================================

@retry(
    stop=stop_after_attempt(RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=RETRY_MIN_WAIT, max=RETRY_MAX_WAIT),
    retry=retry_if_exception_type((
        openai.RateLimitError,
        openai.APITimeoutError,
        openai.APIConnectionError,
    )),
    reraise=True,
)
async def completion(
    client: AsyncOpenAI,
    model: str,
    messages: List[dict],
    max_tokens: int = 200,
    temperature: float = 0.0,
    **kwargs,
) -> str:
    """Single chat completion with retry logic.

    Args:
        client: AsyncOpenAI client
        model: Model name (e.g., "meta-llama/llama-3.1-8b-instruct", "openai/gpt-4.1")
        messages: Chat messages
        max_tokens: Max tokens to generate
        temperature: Sampling temperature
        **kwargs: Additional args passed to API

    Returns:
        Response content string
    """
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        **kwargs,
    )
    return response.choices[0].message.content or ""


# =============================================================================
# Batch Processing
# =============================================================================

async def batch_process(
    items: List[T],
    processor: Callable[[List[T]], Awaitable[List[R]]],
    batch_size: int = 20,
    max_concurrent: int = 50,
) -> List[R]:
    """Process items in batches with semaphore-based concurrency control.

    Args:
        items: Items to process
        processor: Async function that takes a batch and returns list of results
        batch_size: Number of items per batch
        max_concurrent: Maximum concurrent batch calls

    Returns:
        Flattened list of results in same order as input items
    """
    if not items:
        return []

    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_semaphore(batch: List[T]) -> List[R]:
        async with semaphore:
            return await processor(batch)

    batch_results = await asyncio.gather(*[
        process_with_semaphore(batch) for batch in batches
    ])

    return [item for batch_result in batch_results for item in batch_result]


async def batch_completions(
    client: AsyncOpenAI,
    model: str,
    prompts: List[str],
    max_tokens: int = 200,
    temperature: float = 0.0,
    max_concurrent: int = 50,
    system_message: str | None = None,
) -> List[str]:
    """Batch chat completions with concurrency control.

    Args:
        client: AsyncOpenAI client
        model: Model name
        prompts: List of user prompts
        max_tokens: Max tokens per response
        temperature: Sampling temperature
        max_concurrent: Max concurrent API calls
        system_message: Optional system message for all prompts

    Returns:
        List of response strings in same order as prompts
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def complete_with_semaphore(prompt: str) -> str:
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        async with semaphore:
            return await completion(
                client, model, messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

    return list(await asyncio.gather(*[
        complete_with_semaphore(prompt) for prompt in prompts
    ]))


# =============================================================================
# Sync Wrapper
# =============================================================================

def run_sync(async_fn):
    """Decorator to create synchronous wrapper for an async function."""
    def wrapper(*args, **kwargs):
        return asyncio.run(async_fn(*args, **kwargs))
    wrapper.__name__ = f"{async_fn.__name__}_sync"
    wrapper.__doc__ = f"Synchronous wrapper for {async_fn.__name__}"
    return wrapper


# =============================================================================
# Shared Lock Engine
# =============================================================================

class AsyncSharedLockEngine:
    """Async API engine with shared semaphore for rate limiting.

    Use when making multiple API calls that should share a single
    concurrency limit across the workflow (e.g., iterative auditing).

    For batch operations, use asyncio.gather with generate() directly.
    """

    def __init__(self, max_concurrent: int = 50, model: str = "meta-llama/llama-3.1-8b-instruct"):
        if max_concurrent < 1:
            raise ValueError(f"max_concurrent must be >= 1, got {max_concurrent}")
        self.client = get_client()
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.model = model

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

    async def generate_messages(
        self,
        messages: List[dict],
        max_tokens: int = 200,
        temperature: float = 0.0,
        **kwargs,
    ) -> str:
        async with self.semaphore:
            return await completion(
                self.client, self.model,
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.0,
        *,
        system_message: str | None = None,
        **kwargs,
    ) -> str:
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        return await self.generate_messages(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

    async def close(self) -> None:
        await self.client.close()
