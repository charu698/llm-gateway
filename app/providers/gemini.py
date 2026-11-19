import httpx

from app.providers.base import Provider, ProviderError
from app.schemas import ChatRequest, ChatResponse, Choice, ChoiceMessage, Usage

BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiProvider(Provider):
    """Adapter translating OpenAI-style chat requests to Gemini's generateContent API."""

    name = "gemini"

    def __init__(self, api_key: str, timeout: float = 30.0):
        self.api_key = api_key
        self.timeout = timeout

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

    async def chat(self, request: ChatRequest) -> ChatResponse:
        url = f"{BASE_URL}/{request.model}:generateContent"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    url,
                    json=self._to_payload(request),
                    headers={"x-goog-api-key": self.api_key},
                )
            resp.raise_for_status()
            data = resp.json()
            parts = data["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts)
            usage = data.get("usageMetadata", {})
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            raise ProviderError(f"gemini failed: {exc}") from exc

        prompt = usage.get("promptTokenCount", 0)
        completion = usage.get("candidatesTokenCount", 0)
        return ChatResponse(
            model=request.model,
            choices=[Choice(message=ChoiceMessage(content=text))],
            usage=Usage(
                prompt_tokens=prompt,
                completion_tokens=completion,
                total_tokens=prompt + completion,
            ),
        )
