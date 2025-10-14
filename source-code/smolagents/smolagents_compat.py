"""Compatibility tweaks for changes in smolagents dependencies."""

from __future__ import annotations

import inspect
import json
from typing import Dict, List, Optional

import litellm


def _patch_transformers_soundfile() -> None:
    try:
        from transformers import utils as transformers_utils
    except Exception:
        return

    if hasattr(transformers_utils, "is_soundfile_available") and not hasattr(
        transformers_utils, "is_soundfile_availble"
    ):
        transformers_utils.is_soundfile_availble = (
            transformers_utils.is_soundfile_available
        )


def _patch_litellm_model() -> None:
    try:
        from smolagents.models import (
            LiteLLMModel as BaseLiteLLMModel,
            get_clean_message_list,
            get_json_schema,
            tool_role_conversions,
        )
    except Exception:
        return

    if "api_base" in inspect.signature(BaseLiteLLMModel.__init__).parameters:
        return

    original_init = BaseLiteLLMModel.__init__

    def __init__(
        self,
        model_id: str = "anthropic/claude-3-5-sonnet-20240620",
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **_: Dict,
    ) -> None:
        original_init(self, model_id=model_id)
        self._smolagents_api_base = base_url or api_base
        self._smolagents_api_key = api_key

    def __call__(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        max_tokens: int = 1500,
    ) -> str:
        messages = get_clean_message_list(
            messages, role_conversions=tool_role_conversions
        )
        response = litellm.completion(
            model=self.model_id,
            messages=messages,
            stop=stop_sequences,
            max_tokens=max_tokens,
            base_url=self._smolagents_api_base,
            api_key=self._smolagents_api_key,
        )
        self.last_input_token_count = response.usage.prompt_tokens
        self.last_output_token_count = response.usage.completion_tokens
        return response.choices[0].message.content

    def get_tool_call(
        self,
        messages: List[Dict[str, str]],
        available_tools,
        stop_sequences: Optional[List[str]] = None,
        max_tokens: int = 1500,
    ):
        messages = get_clean_message_list(
            messages, role_conversions=tool_role_conversions
        )
        response = litellm.completion(
            model=self.model_id,
            messages=messages,
            tools=[get_json_schema(tool) for tool in available_tools],
            tool_choice="required",
            stop=stop_sequences,
            max_tokens=max_tokens,
            base_url=self._smolagents_api_base,
            api_key=self._smolagents_api_key,
        )
        tool_calls = response.choices[0].message.tool_calls[0]
        self.last_input_token_count = response.usage.prompt_tokens
        self.last_output_token_count = response.usage.completion_tokens
        arguments = json.loads(tool_calls.function.arguments)
        return tool_calls.function.name, arguments, tool_calls.id

    BaseLiteLLMModel.__init__ = __init__
    BaseLiteLLMModel.__call__ = __call__
    BaseLiteLLMModel.get_tool_call = get_tool_call


_patch_transformers_soundfile()
_patch_litellm_model()
