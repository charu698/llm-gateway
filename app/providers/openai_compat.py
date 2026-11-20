import json
from typing import AsyncIterator, Optional

import httpx

from app.providers.base import Provider, ProviderError, StreamEvent, status_error
from app.schemas import ChatRequest, ChatResponse, Choice, ChoiceMessage, Usage


class OpenAICompatProvider(Provider):
    """Any OpenAI-style `/chat/completions` endpoint: Groq, OpenRouter, vLLM, Ollama..."""

    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: str,
        model_override: Optional[str] = None,
        timeout: float = 30.0,
        client: Optional[httpx.AsyncClient] = None,
    ):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_override = model_override or None
        self._client = client or httpx.AsyncClient(timeout=timeout)

    @property
    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"}

    def _payload(self, request: ChatRequest, stream: bool) -> dict:
        payload: dict = {
            "model": self.resolve_model(request),
            "messages": [m.model_dump() for m in request.messages],
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        return payload

    @staticmethod
    def _usage(raw: Optional[dict]) -> Usage:
        raw = raw or {}
        prompt = raw.get("prompt_tokens", 0)
        completion = raw.get("completion_tokens", 0)
        return Usage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=raw.get("total_tokens", prompt + completion),
        )

    async def chat(self, request: ChatRequest) -> ChatResponse:
        url = f"{self.base_url}/chat/completions"
        try:
            resp = await self._client.post(
                url, json=self._payload(request, stream=False), headers=self._headers
            )
        except httpx.HTTPError as exc:
            raise ProviderError(f"{self.name} transport error: {exc!r}", retryable=True) from exc
        if resp.status_code >= 400:
            raise status_error(self.name, resp.status_code, resp.text)
        try:
            data = resp.json()
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, ValueError) as exc:
            raise ProviderError(f"{self.name} returned an unexpected body: {exc!r}") from exc
        return ChatResponse(
            model=self.resolve_model(request),
            choices=[Choice(message=ChoiceMessage(content=text))],
            usage=self._usage(data.get("usage")),
        )

    async def stream(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        url = f"{self.base_url}/chat/completions"
        try:
            async with self._client.stream(
                "POST", url, json=self._payload(request, stream=True), headers=self._headers
            ) as resp:
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode(errors="replace")
                    raise status_error(self.name, resp.status_code, body)
                usage: Optional[Usage] = None
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        data = json.loads(payload)
                    except ValueError:
                        continue
                    for choice in data.get("choices") or []:
                        text = (choice.get("delta") or {}).get("content")
                        if text:
                            yield StreamEvent(delta=text)
                    if data.get("usage"):
                        usage = self._usage(data["usage"])
                if usage:
                    yield StreamEvent(usage=usage)
        except httpx.HTTPError as exc:
            raise ProviderError(f"{self.name} stream error: {exc!r}", retryable=True) from exc

    async def aclose(self) -> None:
        await self._client.aclose()
