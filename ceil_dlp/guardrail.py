"""ceil-dlp as a LiteLLM guardrail.

``ceil_dlp.ceil_dlp_callback`` plugs ceil-dlp into LiteLLM as a plain callback and
has to replace two of LiteLLM's internal functions to restore whistledown tokens
in streamed replies. This module is the alternative for LiteLLM versions that have
the unified guardrail interface (``CustomGuardrail.apply_guardrail``): LiteLLM hands
the guardrail the texts of a request or of a reply, puts changed texts back, and
handles streaming itself, so nothing in LiteLLM is patched.

It uses the same detection, policy file (``CEIL_DLP_CONFIG_PATH``), whistledown
cache and audit log as the callback. Use one or the other, not both.

LiteLLM looks a custom guardrail up as a file next to its config, so put a
one-line module there::

    # ceil_dlp_guardrail.py
    from ceil_dlp.guardrail import CeilDLPGuardrail  # noqa: F401

and register it in the LiteLLM config::

    guardrails:
      - guardrail_name: ceil-dlp
        litellm_params:
          guardrail: ceil_dlp_guardrail.CeilDLPGuardrail
          mode: [pre_call, post_call]
          default_on: true

Differences from the callback:

- ``reasoning_content`` is not rewritten: LiteLLM passes a guardrail the reply's
  content only. Whistledown tokens stay as tokens there, and a streamed reply
  carries no reasoning deltas at all.
- Only text is scanned; images and PDFs are not passed to ``apply_guardrail``.

Validated against LiteLLM 1.108.5.
"""

import os
from typing import Any, Literal

from fastapi import HTTPException
from litellm.integrations.custom_guardrail import CustomGuardrail

from ceil_dlp.middleware import CeilDLPHandler, create_handler
from ceil_dlp.redaction import redact_text
from ceil_dlp.whistledown import whistledown_transform_text


class CeilDLPGuardrail(CustomGuardrail):
    """Apply the ceil-dlp policy to request and response texts."""

    # Streamed replies: LiteLLM passes the reply so far on every chunk and sends
    # on whatever is new in what comes back, so a token or a secret split across
    # chunks is seen whole before any of it is emitted.
    streaming_transform_mode = "incremental_diff"
    streaming_sampling_rate = 1
    # Trailing characters of a streamed reply to withhold until more arrives.
    # Same window as _PatchedCeilDLPHandler._MASK_KEEP: longer than any secret
    # ceil-dlp recognises. The whole tail is released at end of stream.
    STREAM_HOLDBACK_CHARS = 256

    def __init__(self, handler: CeilDLPHandler | None = None, **kwargs: Any) -> None:
        if handler is None:
            path = os.getenv("CEIL_DLP_CONFIG_PATH")
            if path and os.path.isfile(path):
                handler = create_handler(config_path=path)
            else:
                handler = CeilDLPHandler()
        self.dlp = handler
        super().__init__(**kwargs)

    async def apply_guardrail(  # type: ignore[override]
        self,
        inputs: Any,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Any = None,
    ) -> Any:
        request_data = request_data or {}
        model = inputs.get("model") or request_data.get("model") or ""
        request_id = str(request_data.get("litellm_call_id") or "unknown")
        metadata = request_data.get("metadata") or request_data.get("litellm_metadata") or {}
        user_id = metadata.get("user_api_key_user_id")
        texts = inputs.get("texts") or []
        if input_type == "request":
            inputs["texts"] = [self._request_text(t, model, request_id, user_id) for t in texts]
        else:
            inputs["texts"] = [self._response_text(t, model, request_id, user_id) for t in texts]
            inputs["stream_holdback_chars"] = [self.STREAM_HOLDBACK_CHARS] * len(texts)
        return inputs

    def _request_text(self, text: str, model: str, request_id: str, user_id: str | None) -> str:
        if not text:
            return text
        dlp, config = self.dlp, self.dlp.config
        detections, blocked, masked, whistled, *_ = dlp._process_pii_detection(
            [{"role": "user", "content": text}], model
        )
        if not detections:
            return text

        if config.mode == "observe":
            for pii_type, matches in detections.items():
                policy = config.get_policy(pii_type)
                if policy and policy.enabled and dlp._should_apply_policy(policy, model):
                    dlp.audit_logger.log_detection(
                        user_id=user_id,
                        pii_type=pii_type,
                        action="observe",
                        redacted_items=[m[0] for m in matches],
                        request_id=request_id,
                        mode="observe",
                    )
            return text

        if blocked:
            dlp.audit_logger.log_block(
                user_id=user_id, pii_types=blocked, request_id=request_id, mode=config.mode
            )
            # The value is never part of the message, only the kind of data.
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "[ceil-dlp] Request blocked: Detected sensitive data "
                    f"({', '.join(blocked)})"
                },
            )

        for found, action in ((masked, "mask"), (whistled, "whistledown")):
            for pii_type, matches in found.items():
                dlp.audit_logger.log_detection(
                    user_id=user_id,
                    pii_type=pii_type,
                    action=action,
                    redacted_items=[m[0] for m in matches],
                    request_id=request_id,
                    mode=config.mode,
                )

        if whistled:  # first, while the detected positions still match the text
            text, _ = whistledown_transform_text(
                text, detections=whistled, cache=dlp.whistledown_cache, request_id=request_id
            )
        if masked:
            # After a whistledown pass the positions have moved, so detect the
            # masked types again in the new text instead of reusing them.
            text, _ = redact_text(
                text,
                detections=None if whistled else masked,
                ner_strength=config.ner_strength,
                enabled_types=set(masked),
                custom_patterns=config.custom_patterns,
            )
        return text

    def _response_text(self, text: str, model: str, request_id: str, user_id: str | None) -> str:
        if not text or self.dlp.config.mode == "observe":
            return text
        text = self.dlp.mask_response_text(text, model, request_id, user_id)
        return self.dlp.whistledown_cache.reverse_transform(request_id, text)
