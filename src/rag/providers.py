"""Lazy imports for the hosted-provider SDKs.

Both SDKs are optional at runtime: the app works with no key and no network.
Importing them lazily keeps startup fast, and importing them through here
turns a missing package into an instruction rather than a bare
ModuleNotFoundError from deep inside a request.

Callers check for an API key *before* calling these. Someone without a key
needs to be told about the key; the package only matters once they have one.
"""

from __future__ import annotations


def import_genai():
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError(
            "The Google Gemini SDK is not installed. Run: pip install google-genai"
        ) from exc
    return genai, types


def import_openai():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("The OpenAI SDK is not installed. Run: pip install openai") from exc
    return OpenAI
