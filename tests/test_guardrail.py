"""Tests for the LiteLLM guardrail adapter (ceil_dlp.guardrail)."""

import asyncio

import pytest
from fastapi import HTTPException

from ceil_dlp.config import Config, Policy
from ceil_dlp.guardrail import CeilDLPGuardrail
from ceil_dlp.middleware import CeilDLPHandler

API_KEY = "sk-proj-abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKL"
EMAIL = "jane.doe@example.org"
CARD = "4111 1111 1111 1111"


def make(mode: str = "enforce") -> CeilDLPGuardrail:
    config = Config(
        mode=mode,
        ner_strength=1,
        policies={
            "credit_card": Policy(action="block"),
            "api_key": Policy(action="mask"),
            "email": Policy(action="whistledown"),
        },
    )
    return CeilDLPGuardrail(handler=CeilDLPHandler(config=config), guardrail_name="ceil-dlp")


def apply(
    guardrail: CeilDLPGuardrail, text: str, input_type: str = "request", request_id: str = "req-1"
):
    inputs = {"texts": [text], "model": "test-model"}
    data = {"litellm_call_id": request_id, "metadata": {"user_api_key_user_id": "u1"}}
    return asyncio.run(guardrail.apply_guardrail(inputs, data, input_type))  # type: ignore[arg-type]


def test_uses_the_unified_interface_and_transforming_streams():
    assert CeilDLPGuardrail.uses_apply_guardrail_interface()
    assert CeilDLPGuardrail.streaming_transform_mode == "incremental_diff"


def test_clean_text_is_untouched():
    assert apply(make(), "hello there")["texts"] == ["hello there"]


def test_blocked_type_raises_without_the_value():
    with pytest.raises(HTTPException) as raised:
        apply(make(), f"Charge my card {CARD} please")
    assert raised.value.status_code == 400
    message = raised.value.detail["error"]
    assert "[ceil-dlp] Request blocked" in message and "credit_card" in message
    assert CARD not in message


def test_mask_and_whistledown_in_one_text_then_restored_in_the_reply():
    guardrail = make()
    sent = apply(guardrail, f"my email is {EMAIL} and my key is {API_KEY}")["texts"][0]
    assert EMAIL not in sent and API_KEY not in sent
    assert "EMAIL_1" in sent and "[REDACTED_API_KEY]" in sent

    reply = apply(guardrail, "I will write to EMAIL_1 now.", input_type="response")
    assert reply["texts"] == [f"I will write to {EMAIL} now."]
    # A streamed reply keeps its tail back until more text arrives.
    assert reply["stream_holdback_chars"] == [CeilDLPGuardrail.STREAM_HOLDBACK_CHARS]


def test_tokens_are_not_restored_for_another_request():
    guardrail = make()
    apply(guardrail, f"write to {EMAIL}", request_id="req-a")
    other = apply(guardrail, "EMAIL_1 is unknown here", input_type="response", request_id="req-b")
    assert other["texts"] == ["EMAIL_1 is unknown here"]


def test_a_secret_generated_by_the_model_is_masked_in_the_reply():
    reply = apply(make(), f"Use this key: {API_KEY}", input_type="response")["texts"][0]
    assert API_KEY not in reply and "[REDACTED_API_KEY]" in reply


def test_observe_mode_changes_nothing_and_blocks_nothing():
    guardrail = make(mode="observe")
    text = f"card {CARD}, email {EMAIL}, key {API_KEY}"
    assert apply(guardrail, text)["texts"] == [text]
    assert apply(guardrail, "EMAIL_1", input_type="response")["texts"] == ["EMAIL_1"]
