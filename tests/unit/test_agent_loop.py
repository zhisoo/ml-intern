from types import SimpleNamespace

import pytest

from agent.core.agent_loop import (
    _call_llm_streaming,
    _friendly_error_message,
    _is_bedrock_streaming_permission_error,
)


class RecordingSession:
    def __init__(self):
        self.events = []
        self.stream = True

    async def send_event(self, event):
        self.events.append(event)


def test_detects_bedrock_streaming_permission_error():
    error = Exception(
        'litellm.APIConnectionError: BedrockException - {"Message":"User is not '
        "authorized to perform: bedrock:InvokeModelWithResponseStream on "
        'resource: arn:aws:bedrock:us-west-2:123456789012:inference-profile/'
        'us.anthropic.claude-opus-4-6-v1"}'
    )

    assert _is_bedrock_streaming_permission_error(error) is True


def test_friendly_error_message_for_bedrock_streaming_permission():
    error = Exception(
        "BedrockException: not authorized to perform "
        "bedrock:InvokeModelWithResponseStream"
    )

    message = _friendly_error_message(error)

    assert message is not None
    assert "Bedrock denied streaming" in message
    assert "--no-stream" in message


@pytest.mark.asyncio
async def test_streaming_call_falls_back_to_non_streaming_for_bedrock_permission_error(
    monkeypatch,
):
    calls: list[bool] = []

    async def fake_acompletion(*, stream, **_kwargs):
        calls.append(stream)
        if stream:
            raise Exception(
                "BedrockException: User is not authorized to perform "
                "bedrock:InvokeModelWithResponseStream on resource "
                "arn:aws:bedrock:us-west-2:123456789012:inference-profile/"
                "us.anthropic.claude-opus-4-6-v1"
            )

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="fallback response", tool_calls=None),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(total_tokens=17),
        )

    monkeypatch.setattr("agent.core.agent_loop.acompletion", fake_acompletion)

    session = RecordingSession()
    result = await _call_llm_streaming(
        session,
        messages=[],
        tools=[],
        llm_params={"model": "bedrock/us.anthropic.claude-opus-4-6-v1"},
    )

    assert calls == [True, False]
    assert session.stream is False
    assert result.content == "fallback response"
    assert [event.event_type for event in session.events] == [
        "tool_log",
        "assistant_message",
    ]
