from app.providers.base import Provider
from app.schemas import ChatRequest, ChatResponse, Choice, ChoiceMessage, Usage


class MockProvider(Provider):
    """Deterministic offline provider for local dev, tests and demos."""

    name = "mock"

    async def chat(self, request: ChatRequest) -> ChatResponse:
        last_user = next(
            (m.content for m in reversed(request.messages) if m.role == "user"), ""
        )
        reply = f"[mock] You said: {last_user}"
        prompt_tokens = sum(len(m.content.split()) for m in request.messages)
        completion_tokens = len(reply.split())
        return ChatResponse(
            model=request.model,
            choices=[Choice(message=ChoiceMessage(content=reply))],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )
