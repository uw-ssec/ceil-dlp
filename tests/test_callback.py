"""Tests for ceil_dlp_callback module."""

import asyncio
import importlib
import json

from ceil_dlp.ceil_dlp_callback import (
    _build_flush_chunk,
    _chunk_has_finish_reason,
    _PatchedCeilDLPHandler,
    _reorder_stream_for_strict_clients,
    proxy_handler_instance,
)
from ceil_dlp.middleware import CeilDLPHandler


def test_callback_module_default_handler():
    """Test that callback module exports a handler instance."""
    assert proxy_handler_instance is not None
    assert isinstance(proxy_handler_instance._inner, CeilDLPHandler)
    assert isinstance(proxy_handler_instance, CeilDLPHandler) or hasattr(
        proxy_handler_instance, "async_pre_call_hook"
    )


def test_callback_module_with_config_path(tmp_path, monkeypatch):
    """Test callback module with CEIL_DLP_CONFIG_PATH set."""
    # Create a test config file
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(
        "mode: observe\npolicies:\n  email:\n    action: mask\n    enabled: true\n"
    )

    # Set environment variable
    monkeypatch.setenv("CEIL_DLP_CONFIG_PATH", str(config_file))

    # Reload the module to pick up the new env var
    import ceil_dlp.ceil_dlp_callback

    importlib.reload(ceil_dlp.ceil_dlp_callback)

    # Check that handler was created with config
    handler = ceil_dlp.ceil_dlp_callback.proxy_handler_instance
    inner = handler._inner
    assert inner.config.mode == "observe"
    email_policy = inner.config.get_policy("email")
    assert email_policy is not None
    assert email_policy.action == "mask"

    # Clean up - restore original
    monkeypatch.delenv("CEIL_DLP_CONFIG_PATH", raising=False)
    importlib.reload(ceil_dlp.ceil_dlp_callback)


def test_callback_module_with_invalid_config_path(monkeypatch):
    """Test callback module with invalid CEIL_DLP_CONFIG_PATH."""
    # Set environment variable to non-existent file
    monkeypatch.setenv("CEIL_DLP_CONFIG_PATH", "/nonexistent/path.yaml")

    # Reload the module
    import ceil_dlp.ceil_dlp_callback

    importlib.reload(ceil_dlp.ceil_dlp_callback)

    # Should fall back to default handler
    handler = ceil_dlp.ceil_dlp_callback.proxy_handler_instance
    assert isinstance(handler._inner, CeilDLPHandler)
    # Should use default config (enforce mode)
    assert handler._inner.config.mode == "enforce"

    # Clean up
    monkeypatch.delenv("CEIL_DLP_CONFIG_PATH", raising=False)
    importlib.reload(ceil_dlp.ceil_dlp_callback)


def test_chunk_has_finish_reason():
    """Detect terminal chunks in serialized, dict, and object forms."""
    assert _chunk_has_finish_reason('{"choices":[{"finish_reason":"stop"}]}')
    assert _chunk_has_finish_reason('{"choices":[{"finish_reason":"length"}]}')
    assert not _chunk_has_finish_reason('{"choices":[{"finish_reason":null}]}')
    assert not _chunk_has_finish_reason('{"choices":[{"delta":{"content":"hi"}}]}')
    assert _chunk_has_finish_reason({"choices": [{"finish_reason": "stop"}]})
    assert not _chunk_has_finish_reason({"choices": [{"delta": {}}]})
    assert not _chunk_has_finish_reason("not json")


def test_build_flush_chunk_is_valid_streaming_chunk():
    """The residual flush must be a normal stream delta, not a full completion."""
    chunk = json.loads(_build_flush_chunk('{"content": "tail"}', model="test-model"))
    assert chunk["object"] == "chat.completion.chunk"
    assert chunk["model"] == "test-model"
    assert chunk["choices"][0]["delta"] == {"content": "tail"}
    assert chunk["choices"][0]["finish_reason"] is None
    # A full completion object here would be mistaken for a non-stream payload.
    assert "message" not in chunk["choices"][0]


def test_flush_keeps_reasoning_out_of_the_answer():
    """A reasoning model's held-back thinking must not be flushed as content."""
    handler = _PatchedCeilDLPHandler(CeilDLPHandler())
    handler._stream_buffers["req-1:content"] = "the answer"
    handler._stream_buffers["req-1:reasoning"] = "the thinking"

    residual = handler.flush_streaming_buffers("req-1")
    assert json.loads(residual) == {"reasoning_content": "the thinking", "content": "the answer"}
    delta = json.loads(_build_flush_chunk(residual))["choices"][0]["delta"]
    assert delta["content"] == "the answer"
    assert delta["reasoning_content"] == "the thinking"

    # Nothing buffered: no flush chunk at all.
    assert handler.flush_streaming_buffers("req-1") == ""

    handler._stream_buffers["req-2:reasoning"] = "only thinking"
    delta = json.loads(_build_flush_chunk(handler.flush_streaming_buffers("req-2")))["choices"][0]["delta"]
    assert delta == {"reasoning_content": "only thinking"}


def test_reorder_puts_flush_before_finish_reason():
    """No content may be emitted after a finish_reason chunk."""
    content = '{"choices":[{"delta":{"content":"Hello"}}]}'
    finish = '{"choices":[{"delta":{},"finish_reason":"stop"}]}'
    usage = '{"choices":[{"delta":{}}],"usage":{"total_tokens":5}}'

    async def source():
        yield content
        yield finish
        yield usage

    async def scenario():
        return [
            chunk
            async for chunk in _reorder_stream_for_strict_clients(
                source(),
                lambda: " tail",
                lambda text: json.dumps(
                    {"choices": [{"delta": {"content": text}, "finish_reason": None}]}
                ),
            )
        ]

    out = asyncio.run(scenario())

    # Content first, then any non-terminal chunk, then the flush, then finish.
    assert out[0] == content
    assert out[1] == usage
    assert json.loads(out[2])["choices"][0]["delta"]["content"] == " tail"
    assert out[3] == finish

    seen_finish = False
    for serialized in out:
        choice = json.loads(serialized)["choices"][0]
        assert not (seen_finish and choice.get("delta", {}).get("content"))
        if choice.get("finish_reason"):
            seen_finish = True
    assert seen_finish


def test_reorder_keeps_finish_terminal_without_residual():
    """With nothing buffered the finish_reason chunk still comes last."""

    async def source():
        yield '{"choices":[{"delta":{"content":"Hi"}}]}'
        yield '{"choices":[{"delta":{},"finish_reason":"stop"}]}'

    async def scenario():
        return [
            chunk
            async for chunk in _reorder_stream_for_strict_clients(
                source(), lambda: "", lambda text: text
            )
        ]

    out = asyncio.run(scenario())
    assert _chunk_has_finish_reason(out[-1])
    assert not _chunk_has_finish_reason(out[0])