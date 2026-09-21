"""LLM backends behind one interface.

``gemini`` calls Google's free-tier API. ``mock`` is a deterministic stub that
answers from the supplied context by sentence overlap - no network, no key, no
quota. Every test and the whole CI run uses the mock, which is what keeps the
suite fast and reproducible.
"""

from __future__ import annotations

import logging
import random
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .config import Settings

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LLMResponse:
    text: str
    usage: dict[str, int] = field(default_factory=dict)
    model: str = ""


class LLMClient(ABC):
    name: str

    @abstractmethod
    def generate(self, system: str, prompt: str) -> LLMResponse: ...


class MockLLM(LLMClient):
    """Extractive stand-in for a real model.

    It picks the context sentences that share the most content words with the
    question and cites the passages it drew them from. That is enough to
    exercise prompt assembly, citation parsing and abstention end to end
    without a network call.
    """

    _STOPWORDS = {
        "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
        "for", "from", "how", "in", "is", "it", "of", "on", "or", "that",
        "the", "to", "was", "were", "what", "when", "where", "which", "who",
        "why", "with",
    }

    def __init__(self, min_overlap: int = 2) -> None:
        self.name = "mock"
        self.min_overlap = min_overlap

    def _content_words(self, text: str) -> set[str]:
        return {
            w for w in re.findall(r"[a-z0-9]+", text.lower())
            if w not in self._STOPWORDS and len(w) > 2
        }

    def generate(self, system: str, prompt: str) -> LLMResponse:
        question, passages = _parse_prompt(prompt)
        question_words = self._content_words(question)

        best: list[tuple[int, int, str]] = []
        for marker, passage in passages:
            for sentence in re.split(r"(?<=[.!?])\s+", passage):
                overlap = len(question_words & self._content_words(sentence))
                if overlap >= self.min_overlap:
                    best.append((overlap, marker, sentence.strip()))

        if not best:
            return LLMResponse(text="INSUFFICIENT_CONTEXT", model=self.name)

        best.sort(key=lambda item: (-item[0], item[1]))
        chosen = best[:2]
        answer = " ".join(f"{sentence} [{marker}]" for _, marker, sentence in chosen)
        return LLMResponse(
            text=answer,
            usage={"prompt_tokens": len(prompt.split()), "output_tokens": len(answer.split())},
            model=self.name,
        )


class GeminiLLM(LLMClient):
    def __init__(self, settings: Settings) -> None:
        from google import genai
        from google.genai import types

        self._settings = settings
        self._client = genai.Client(
            api_key=settings.require_api_key(),
            # One retry layer only. The SDK retries internally by default,
            # which would compose with the backoff below into 25 attempts
            # and a request that looks like a hang.
            http_options=types.HttpOptions(
                timeout=settings.request_timeout_ms,
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )
        self._model = settings.gemini_chat_model
        self.name = self._model

    def generate(self, system: str, prompt: str) -> LLMResponse:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=self._settings.temperature,
            max_output_tokens=self._settings.max_output_tokens,
        )
        response = self._call_with_retry(prompt, config)
        usage = {}
        if getattr(response, "usage_metadata", None):
            usage = {
                "prompt_tokens": response.usage_metadata.prompt_token_count or 0,
                "output_tokens": response.usage_metadata.candidates_token_count or 0,
            }
        return LLMResponse(text=(response.text or "").strip(), usage=usage, model=self._model)

    def _call_with_retry(self, prompt: str, config):
        settings = self._settings
        for attempt in range(settings.max_retries):
            try:
                return self._client.models.generate_content(
                    model=self._model, contents=prompt, config=config
                )
            except Exception as exc:
                if not is_retryable(exc) or attempt == settings.max_retries - 1:
                    raise
                delay = settings.retry_base_delay * (2**attempt)
                delay += random.uniform(0, delay * 0.1)  # jitter avoids lockstep retries
                logger.warning("Gemini call failed (%s); retrying in %.1fs", exc, delay)
                time.sleep(delay)
        raise RuntimeError("Unreachable retry state")


class OpenAICompatibleLLM(LLMClient):
    """Any OpenAI-format chat-completions endpoint.

    OpenAI, Groq, Together, OpenRouter and a local Ollama or vLLM all serve
    this same wire format, so one adapter plus a `base_url` covers all of
    them. Keeping generation behind `LLMClient` is what makes that a config
    change rather than a rewrite.
    """

    def __init__(self, settings: Settings) -> None:
        from openai import OpenAI

        self._settings = settings
        self._client = OpenAI(
            api_key=settings.require_openai_key(),
            base_url=settings.openai_base_url or None,
            timeout=settings.request_timeout_ms / 1000,
            # One retry layer only - the backoff below is this module's.
            max_retries=0,
        )
        self._model = settings.openai_chat_model
        self.name = self._model

    def generate(self, system: str, prompt: str) -> LLMResponse:
        response = self._call_with_retry(
            {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "temperature": self._settings.temperature,
                "max_tokens": self._settings.max_output_tokens,
            }
        )
        message = response.choices[0].message if response.choices else None
        usage = {}
        if getattr(response, "usage", None):
            usage = {
                "prompt_tokens": response.usage.prompt_tokens or 0,
                "output_tokens": response.usage.completion_tokens or 0,
            }
        return LLMResponse(
            text=((getattr(message, "content", None) or "").strip()),
            usage=usage,
            model=self._model,
        )

    def _call_with_retry(self, kwargs):
        settings = self._settings
        for attempt in range(settings.max_retries):
            try:
                return self._client.chat.completions.create(**kwargs)
            except Exception as exc:
                if not is_retryable(exc) or attempt == settings.max_retries - 1:
                    raise
                delay = settings.retry_base_delay * (2**attempt)
                delay += random.uniform(0, delay * 0.1)  # jitter avoids lockstep retries
                logger.warning("OpenAI call failed (%s); retrying in %.1fs", exc, delay)
                time.sleep(delay)
        raise RuntimeError("Unreachable retry state")


_RETRYABLE_MARKERS = (
    "429", "rate limit", "resource_exhausted", "503", "500",
    "unavailable", "deadline", "internal error",
)


def is_retryable(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in _RETRYABLE_MARKERS)


_PASSAGE_RE = re.compile(r"^\[(\d+)\]", re.MULTILINE)


def _parse_prompt(prompt: str) -> tuple[str, list[tuple[int, str]]]:
    """Pull the question and the numbered passages back out of a built prompt.

    Only the mock needs this; it lets the stub behave like a model that
    actually read the context instead of echoing a canned string.
    """
    question = ""
    question_match = re.search(r"Question:\s*(.+)", prompt)
    if question_match:
        question = question_match.group(1).strip()

    passages: list[tuple[int, str]] = []
    matches = list(_PASSAGE_RE.finditer(prompt))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(prompt)
        block = prompt[match.end() : end].strip()
        # The rest of the marker line is the "title - source" header, not
        # passage text; quoting it back as an answer would be nonsense.
        _, _, body = block.partition("\n")
        passages.append((int(match.group(1)), body.strip()))
    return question, passages


def get_llm(settings: Settings) -> LLMClient:
    backend = settings.llm_backend
    if backend == "gemini":
        return GeminiLLM(settings)
    if backend == "openai":
        return OpenAICompatibleLLM(settings)
    if backend == "mock":
        return MockLLM()
    raise ValueError(f"Unknown LLM backend: {backend!r}")
