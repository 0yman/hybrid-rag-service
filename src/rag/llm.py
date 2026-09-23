"""Answer engines behind one interface.

``gemini`` and ``openai`` call hosted models. ``extractive`` needs no model at
all: it quotes the sentences from the retrieved passages that best match the
question, with citations. It is the default when no API key is configured, so
the app answers something useful the moment it is installed - and because it is
deterministic, the test suite and CI run on it too.
"""

from __future__ import annotations

import logging
import random
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .config import Settings
from .providers import import_genai, import_openai

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


class ExtractiveLLM(LLMClient):
    """Answers by quoting, not generating.

    Picks the sentences in the retrieved passages closest to the question and
    cites the passage each came from. It cannot paraphrase or combine facts,
    but it also cannot invent one: every word it returns is in your documents.
    That makes it an honest zero-cost default and a useful baseline to compare
    a hosted model against.

    Two ways to judge "closest":

    * **By meaning** (when given an embedder): cosine similarity between the
      question and each sentence. This is the default in the app. On the
      benchmark (fastembed runtime) it found more of the expected facts than
      matching by words (0.790 vs 0.726), answered every paraphrased question,
      and declined every off-topic one (1.000 vs 0.875).
    * **By shared words** (no embedder): the fallback, and what the test suite
      uses. It is deterministic without a model, but it misses paraphrases -
      "How long does the warranty last?" shares one word with "The warranty
      covers defects for twenty four months", which is not enough.
    """

    _STOPWORDS = {
        "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
        "for", "from", "how", "in", "is", "it", "of", "on", "or", "that",
        "the", "to", "was", "were", "what", "when", "where", "which", "who",
        "why", "with",
    }

    def __init__(
        self,
        embedder=None,
        min_similarity: float = 0.45,
        min_overlap: int = 2,
        max_sentences: int = 3,
    ) -> None:
        self.name = "extractive"
        self.embedder = embedder
        self.min_similarity = min_similarity
        self.min_overlap = min_overlap
        self.max_sentences = max_sentences

    def _content_words(self, text: str) -> set[str]:
        return {
            w for w in re.findall(r"[a-z0-9]+", text.lower())
            if w not in self._STOPWORDS and len(w) > 2
        }

    def _candidates(self, passages: list[tuple[int, str, str]]) -> list[tuple[int, str]]:
        found: list[tuple[int, str]] = []
        seen: set[str] = set()
        for marker, title, passage in passages:
            for sentence in re.split(r"(?<=[.!?])\s+", _strip_headings(passage, title)):
                sentence = " ".join(sentence.split())  # no stray newlines in a quote
                # Chunks overlap by design, so the same sentence often appears
                # in two neighbouring passages. Quote it once.
                key = " ".join(sentence.lower().split())
                if sentence and key not in seen:
                    seen.add(key)
                    found.append((marker, sentence))
        return found

    def _by_meaning(self, question: str, candidates: list[tuple[int, str]]) -> list[tuple[int, str]]:
        import numpy as np

        query = self.embedder.embed_query(question)
        sims = self.embedder.embed_documents([s for _, s in candidates]) @ query
        order = np.argsort(-sims)
        best = float(sims[order[0]])
        if best < self.min_similarity:
            return []
        # Keep only sentences near the best one, so a strong match is not
        # padded out with weak ones just to reach max_sentences.
        floor = max(self.min_similarity, best - 0.15)
        return [candidates[i] for i in order[: self.max_sentences] if sims[i] >= floor]

    def _by_words(self, question: str, candidates: list[tuple[int, str]]) -> list[tuple[int, str]]:
        question_words = self._content_words(question)
        scored = [
            (len(question_words & self._content_words(sentence)), marker, sentence)
            for marker, sentence in candidates
        ]
        scored = [item for item in scored if item[0] >= self.min_overlap]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [(marker, sentence) for _, marker, sentence in scored[: self.max_sentences]]

    def generate(self, system: str, prompt: str) -> LLMResponse:
        question, passages = _parse_prompt(prompt)
        candidates = self._candidates(passages)
        if not candidates:
            return LLMResponse(text="INSUFFICIENT_CONTEXT", model=self.name)

        chosen = (
            self._by_meaning(question, candidates)
            if self.embedder is not None
            else self._by_words(question, candidates)
        )
        if not chosen:
            return LLMResponse(text="INSUFFICIENT_CONTEXT", model=self.name)

        answer = " ".join(f"{sentence} [{marker}]" for marker, sentence in chosen)
        return LLMResponse(
            text=answer,
            usage={"prompt_tokens": len(prompt.split()), "output_tokens": len(answer.split())},
            model=self.name,
        )


# A heading runs from its marker to the end of its line. The chunker always
# keeps that newline (see chunking.split_paragraphs), so the heading's end is
# knowable wherever it sits - even mid-chunk, after another sentence.
_HEADING_LINE = re.compile(r"#{1,6}[ \t]+[^\n]*\n")
_HEADING_MARK = re.compile(r"#{1,6}\s+")


def _strip_headings(passage: str, title: str) -> str:
    """Remove markdown headings so they are never quoted as part of an answer.

    Chunks keep headings because they help retrieval, but "## Shipping In
    commercial chartering..." reads as garbage when quoted back to a person.
    """
    if title:
        passage = re.sub(rf"^\s*#{{1,6}}\s*{re.escape(title)}\s*", "", passage)
    passage = _HEADING_LINE.sub(" ", passage)
    return _HEADING_MARK.sub("", passage)


class GeminiLLM(LLMClient):
    def __init__(self, settings: Settings) -> None:
        # The key first: someone without one needs to hear about the key, not
        # about a package they would only need once they had one.
        api_key = settings.require_api_key()
        genai, types = import_genai()

        self._settings = settings
        self._client = genai.Client(
            api_key=api_key,
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
        api_key = settings.require_openai_key()  # before the import; see GeminiLLM
        OpenAI = import_openai()  # noqa: N806 - it is a class

        self._settings = settings
        self._client = OpenAI(
            api_key=api_key,
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


def _parse_prompt(prompt: str) -> tuple[str, list[tuple[int, str, str]]]:
    """Pull the question and the numbered passages back out of a built prompt.

    Only the extractive engine needs this: it reads the same prompt a hosted
    model would, so both are fed by exactly the same retrieval.
    """
    question = ""
    question_match = re.search(r"Question:\s*(.+)", prompt)
    if question_match:
        question = question_match.group(1).strip()

    passages: list[tuple[int, str, str]] = []
    matches = list(_PASSAGE_RE.finditer(prompt))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(prompt)
        block = prompt[match.end() : end].strip()
        # The rest of the marker line is the "title - source" header, not
        # passage text; quoting it back as an answer would be nonsense.
        header, _, body = block.partition("\n")
        title = header.rsplit(" - ", 1)[0].strip()
        passages.append((int(match.group(1)), title, body.strip()))
    return question, passages


def get_llm(settings: Settings) -> LLMClient:
    backend = settings.resolved_llm_backend()
    if backend == "gemini":
        return GeminiLLM(settings)
    if backend == "openai":
        return OpenAICompatibleLLM(settings)
    if backend == "extractive":
        return ExtractiveLLM()
    raise ValueError(f"Unknown LLM backend: {backend!r}")
