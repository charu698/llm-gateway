import json
from typing import AsyncIterator, Optional

import httpx

from app.providers.base import Provider, ProviderError, StreamEvent, status_error
from app.schemas import ChatRequest, ChatResponse, Choice, ChoiceMessage, Usage

BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


def _text_of(data: dict) -> str:
    parts = data["candidates"][0]["content"]["parts"]
    return "".join(p.get("text", "") for p in parts)


def _usage_of(meta: dict) -> Usage:
    prompt = meta.get("promptTokenCount", 0)
    completion = meta.get("candidatesTokenCount", 0)
    return Usage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
    )


class GeminiProvider(Provider):
    """Adapter translating OpenAI-style chat requests to Gemini's generateContent API."""

    name = "gemini"

    def __init__(
        self,
        api_key: str,
        timeout: float = 30.0,
        model_override: Optional[str] = None,
        client: Optional[httpx.AsyncClient] = None,
        base_url: str = BASE_URL,
    ):
        self.api_key = api_key
        self.model_override = model_override or None
        self.base_url = base_url
        self._client = client or httpx.AsyncClient(timeout=timeout)

    @staticmethod
    def _to_payload(request: ChatRequest) -> dict:
        system = [m.content for m in request.messages if m.role == "system"]
        contents = [
            {
                "role": "model" if m.role == "assistant" else "user",
                "parts": [{"text": m.content}],
            }
            for m in request.messages
            if m.role != "system"
        ]
        payload: dict = {"contents": contents}
        if system:
            payload["systemInstruction"] = {"parts": [{"text": "\n".join(system)}]}
        gen_config = {}
        if request.temperature is not None:
            gen_config["temperature"] = request.temperature
        if request.max_tokens is not None:
            gen_config["maxOutputTokens"] = request.max_tokens
        if gen_config:
            payload["generationConfig"] = gen_config
        return payload

    @property
    def _headers(self) -> dict:
        return {"x-goog-api-key": self.api_key}

    async def chat(self, request: ChatRequest) -> ChatResponse:
        model = self.resolve_model(request)
        url = f"{self.base_url}/{model}:generateContent"
        try:
            resp = await self._client.post(
                url, json=self._to_payload(request), headers=self._headers
            )
        except httpx.HTTPError as exc:  # timeouts, connection resets, DNS...
            raise ProviderError(f"gemini transport error: {exc!r}", retryable=True) from exc
        if resp.status_code >= 400:
            raise status_error("gemini", resp.status_code, resp.text)
        try:
            data = resp.json()
            text = _text_of(data)
        except (KeyError, IndexError, ValueError) as exc:
            # e.g. a safety block returns 200 with no candidates: not worth retrying
            raise ProviderError(f"gemini returned no usable candidate: {exc!r}") from exc
        return ChatResponse(
            model=model,
            choices=[Choice(message=ChoiceMessage(content=text))],
            usage=_usage_of(data.get("usageMetadata", {})),
        )

    async def stream(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        model = self.resolve_model(request)
        url = f"{self.base_url}/{model}:streamGenerateContent?alt=sse"
        try:
            async with self._client.stream(
                "POST", url, json=self._to_payload(request), headers=self._headers
            ) as resp:
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode(errors="replace")
                    raise status_error("gemini", resp.status_code, body)
                usage: Optional[Usage] = None
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    try:
                        data = json.loads(line[5:].strip())
                    except ValueError:
                        continue
                    try:
                        text = _text_of(data)
                    except (KeyError, IndexError):
                        text = ""
                    if text:
                        yield StreamEvent(delta=text)
                    if data.get("usageMetadata"):
                        usage = _usage_of(data["usageMetadata"])
                if usage:
                    yield StreamEvent(usage=usage)
        except httpx.HTTPError as exc:
            raise ProviderError(f"gemini stream error: {exc!r}", retryable=True) from exc

    async def aclose(self) -> None:
        await self._client.aclose()
