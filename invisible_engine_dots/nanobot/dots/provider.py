"""The Dot's one model provider: OpenRouter, with the key the host pushed to memory.

A provider object holds its key, so a new one is built only when the key, the
model or the base URL changes. A turn captures its provider (and the model it was
built for) when it starts and keeps it: a key or config push builds the next
turn's provider and never swaps the one a running turn uses.
"""

from __future__ import annotations

from nanobot.dots.projection import EngineSettings
from nanobot.providers.openai_compat_provider import OpenAICompatProvider
from nanobot.providers.registry import OPENROUTER, PROVIDERS


class OpenRouterProviders:
    """Hands out a configured OpenAI-compatible provider for the current Dot."""

    def __init__(self) -> None:
        self._identity: tuple[str, str, str | None] | None = None
        self._provider: OpenAICompatProvider | None = None

    def current(self, settings: EngineSettings, key: str) -> OpenAICompatProvider:
        """The provider for `key`, the model and the base URL of `settings`; the same object while they hold."""
        spec = PROVIDERS.get(settings.provider_name)
        if spec is None:
            raise ValueError(f"unsupported provider: {settings.provider_name}")
        base_url = settings.openrouter_base_url if spec is OPENROUTER else None
        identity = (key, settings.provider_name, settings.model_id, base_url)
        if self._provider is None or identity != self._identity:
            self._provider = OpenAICompatProvider(
                api_key=key,
                api_base=base_url or spec.default_api_base,
                default_model=settings.model_id,
                spec=spec,
                provider_name=spec.name,
            )
            self._identity = identity
        return self._provider
