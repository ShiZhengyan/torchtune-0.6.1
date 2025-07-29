# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
import torch
import numpy as np
from typing import Any, Dict, Mapping, Optional, Callable, Union

from torchtune.data._common import CROSS_ENTROPY_IGNORE_IDX
from torchtune.data._messages import OpenAIToMessages, ShareGPTToMessages
from torchtune.datasets._packed import PackedDataset
from torchtune.datasets._sft import SFTDataset, SFTTransform
from torchtune.modules.transforms import Transform
from torchtune.modules.transforms.tokenizers import ModelTokenizer


def classify_tokens_for_tool_calling(
    text: str,
    input_ids: torch.Tensor,
    tokenizer,
    labels: torch.Tensor = None,
    tool_call_start_token: str = "<function",
    tool_call_end_token: str = "</function>",
) -> Dict[str, torch.Tensor]:
    """
    Classify assistant tokens into reasoning / tool-call / special categories.

    A token is tagged as TOOL_CALL iff any part of its character span overlaps a
    tool call block in the decoded text (inclusive of the tags). The tool call
    block is defined by the configurable start and end tokens.
    Special tokens and tokens whose label equals `CROSS_ENTROPY_IGNORE_IDX` are
    excluded from both categories.
    
    Args:
        text: The decoded text
        input_ids: Input token IDs
        tokenizer: The tokenizer
        labels: Optional labels tensor
        tool_call_start_token: Start token for tool calls (default: "<function")
        tool_call_end_token: End token for tool calls (default: "</function>")
    """
    seq_len = len(input_ids)

    # Default: everything is reasoning
    reasoning_mask = torch.ones(seq_len, dtype=torch.bool)
    tool_call_mask = torch.zeros(seq_len, dtype=torch.bool)

    # Identify special tokens once
    special_token_ids = get_special_token_ids(tokenizer)

    # Locate all function blocks in *text* using configurable tokens
    # Escape special regex characters in the tokens
    start_pattern = re.escape(tool_call_start_token)
    end_pattern = re.escape(tool_call_end_token)
    
    # Build regex pattern dynamically
    # Handle case where start token might be incomplete (like "<function" without closing >)
    if tool_call_start_token.endswith(">"):
        pattern = f"{start_pattern}.*?{end_pattern}"
    else:
        # Assume start token needs completion (e.g., "<function" -> "<function[^>]*?>")
        pattern = f"{start_pattern}[^>]*?>.*?{end_pattern}"
    
    block_spans = [
        (m.start(), m.end())
        for m in re.finditer(pattern, text, flags=re.DOTALL)
    ]

    # Pre-compute per-token character spans in *text*
    token_spans = []
    cur = 0
    for tid in input_ids.tolist():
        tok_str = tokenizer.decode([tid], skip_special_tokens=False)
        token_spans.append((cur, cur + len(tok_str)))
        cur += len(tok_str)

    # Helper to know if a span overlaps any function block
    def overlaps_function(span):
        s, e = span
        for bs, be in block_spans:
            if s < be and e > bs:  # any overlap
                return True
        return False

    # Pre-compute tool call start/end token IDs to handle special token exceptions
    tool_call_start_ids = set()
    tool_call_end_ids = set()
    try:
        # Get token IDs for tool call start/end tokens
        if hasattr(tokenizer, 'encode'):
            start_tokens = tokenizer.encode(tool_call_start_token, add_bos=False, add_eos=False)
            end_tokens = tokenizer.encode(tool_call_end_token, add_bos=False, add_eos=False)
            tool_call_start_ids.update(start_tokens)
            tool_call_end_ids.update(end_tokens)
    except Exception:
        # If encoding fails, we'll rely on text-based overlap detection
        pass
    
    # Classify each token
    for idx, tid in enumerate(input_ids.tolist()):
        # Check if this token overlaps with a function block first
        token_overlaps_function = overlaps_function(token_spans[idx])
        
        # If token overlaps with function block, mark as tool call even if it's special
        if token_overlaps_function:
            reasoning_mask[idx] = False
            tool_call_mask[idx] = True
            continue
            
        # Exclude special tokens (but only if they don't overlap with function blocks)
        if tid in special_token_ids:
            reasoning_mask[idx] = False
            continue

        # Exclude ignored labels
        if labels is not None and labels[idx] == CROSS_ENTROPY_IGNORE_IDX:
            reasoning_mask[idx] = False
            continue

    return {
        "reasoning_mask": reasoning_mask,
        "tool_call_mask": tool_call_mask,
    }


def get_special_token_ids(tokenizer):
    """Get special token IDs using tokenizer's native methods."""
    special_ids = set()
    
    # TorchTune tokenizer: use special_tokens dict
    if hasattr(tokenizer, 'special_tokens') and isinstance(tokenizer.special_tokens, dict):
        special_ids.update(tokenizer.special_tokens.values())
    
    # HuggingFace tokenizer: use all_special_ids
    elif hasattr(tokenizer, 'all_special_ids'):
        special_ids.update(tokenizer.all_special_ids)
    
    # Add common special token IDs as fallback, but only if they seem reasonable
    # Avoid adding very low token IDs (0-10) that might be regular vocabulary
    if hasattr(tokenizer, 'pad_id') and tokenizer.pad_id is not None:
        special_ids.add(tokenizer.pad_id)
    if hasattr(tokenizer, 'eos_id') and tokenizer.eos_id is not None and tokenizer.eos_id > 100:
        special_ids.add(tokenizer.eos_id)
    if hasattr(tokenizer, 'bos_id') and tokenizer.bos_id is not None and tokenizer.bos_id > 100:
        special_ids.add(tokenizer.bos_id)
    if hasattr(tokenizer, 'unk_id') and tokenizer.unk_id is not None and tokenizer.unk_id > 100:
        special_ids.add(tokenizer.unk_id)
    
    return special_ids


def find_tool_call_token_regions(
    input_ids, 
    tokenizer, 
    special_token_ids, 
    estimated_tool_tokens,
    tool_call_start_token: str = "<function",
    tool_call_end_token: str = "</function>",
):
    """
    Find token regions that likely correspond to tool calls using simple heuristics.
    
    This is a simplified approach that works reasonably well without complex text reconstruction.
    
    Args:
        input_ids: Input token IDs
        tokenizer: The tokenizer
        special_token_ids: Set of special token IDs
        estimated_tool_tokens: Estimated number of tool tokens
        tool_call_start_token: Start token for tool calls (default: "<function")
        tool_call_end_token: End token for tool calls (default: "</function>")
    """
    regions = []
    seq_len = len(input_ids)
    
    if estimated_tool_tokens <= 0:
        return regions
    
    try:
        # Strategy: Look for known function-related tokens if they exist in the tokenizer
        function_start_tokens = []
        function_end_tokens = []
        
        # Check for common function call markers
        try:
            # Try to find tokens for common function patterns
            if hasattr(tokenizer, 'encode'):
                # Look for function start patterns using configurable token
                try:
                    tokens = tokenizer.encode(tool_call_start_token, add_bos=False, add_eos=False)
                    if tokens:
                        function_start_tokens.extend(tokens)
                except Exception as encode_error:
                    raise RuntimeError(f"Failed to encode function start pattern '{tool_call_start_token}': {type(encode_error).__name__}: {encode_error}") from encode_error
                
                # Look for function end patterns using configurable token
                try:
                    tokens = tokenizer.encode(tool_call_end_token, add_bos=False, add_eos=False)
                    if tokens:
                        function_end_tokens.extend(tokens)
                except Exception as encode_error:
                    raise RuntimeError(f"Failed to encode function end pattern '{tool_call_end_token}': {type(encode_error).__name__}: {encode_error}") from encode_error
        except Exception as tokenizer_error:
            raise RuntimeError(f"Tokenizer access failed in find_tool_call_token_regions: {type(tokenizer_error).__name__}: {tokenizer_error}") from tokenizer_error
        
        # If we found function-related tokens, use them to identify regions
        if function_start_tokens or function_end_tokens:
            start_markers = set(function_start_tokens)
            end_markers = set(function_end_tokens)
            
            current_start = None
            for i, token_id in enumerate(input_ids.tolist()):
                if token_id in start_markers and current_start is None:
                    current_start = i
                elif token_id in end_markers and current_start is not None:
                    # Since </function> is always complete, just include up to the end marker
                    regions.append((current_start, i + 1))
                    current_start = None
            
            # If we have an unclosed region, close it at the end
            if current_start is not None:
                regions.append((current_start, seq_len))
        
        # Fallback: If no specific markers found, estimate based on sequence position
        if not regions and estimated_tool_tokens > 0:
            # Simple heuristic: assume tool calls are in the middle-to-end portion
            # This works for typical agent conversations where reasoning comes first
            start_idx = max(0, seq_len // 3)  # Start after first third
            end_idx = min(seq_len, start_idx + estimated_tool_tokens)
            if end_idx > start_idx:
                regions.append((start_idx, end_idx))
        
    except Exception as region_error:
        raise RuntimeError(f"Function region detection failed: {type(region_error).__name__}: {region_error}") from region_error
    
    return regions


class AgentSFTDataset:
    """
    Enhanced SFT Dataset for agent training with detailed metrics support.
    
    This dataset follows the same simple configuration pattern as chat_dataset
    but adds token classification for reasoning vs tool calling to enable 
    detailed metrics tracking during training.
    """
    
    def __init__(
        self,
        tokenizer: ModelTokenizer,
        *,
        source: str,
        conversation_column: str,
        conversation_style: str,
        train_on_input: bool = False,
        new_system_prompt: Optional[str] = None,
        packed: bool = False,
        filter_fn: Optional[Callable] = None,
        split: str = "train",
        enable_token_classification: bool = True,
        train_on_tool_calls_only: bool = False,       # <── NEW ARG
        tool_call_start_token: str = "<function",
        tool_call_end_token: str = "</function>",
        **load_dataset_kwargs: Dict[str, Any],
    ):
        """
        Args:
            tokenizer (ModelTokenizer): Tokenizer used by the model
            source (str): path to dataset repository on Hugging Face. For local datasets,
                define source as the data file type (e.g. "json", "csv", "text") and pass
                in the filepath in ``data_files``.
            conversation_column (str): name of column containing the conversations
            conversation_style (str): string specifying expected format of conversations in
                the dataset for automatic conversion to the :class:`~torchtune.data.Message`
                format. Supported styles are: "sharegpt" and "openai"
            train_on_input (bool): whether the model is trained on the prompt or not.
                Default is False.
            new_system_prompt (Optional[str]): if specified, prepend a system message. This can
                serve as instructions to guide the model response. Default is None.
            packed (bool): whether or not to pack the dataset to ``max_seq_len`` prior to training.
                Default is False.
            filter_fn (Optional[Callable]): callable used to filter the dataset prior to any pre-processing.
            split (str): ``split`` argument for ``datasets.load_dataset``. You can use this argument to load a subset
                of a given split, e.g. ``split="train[:10%]"``. Default is "train".
            enable_token_classification (bool): whether to enable token classification for agent metrics.
                Default is True.
            train_on_tool_calls_only (bool): whether to replace labels of non-tool-call tokens with
                `CROSS_ENTROPY_IGNORE_IDX`. Default is True.
            tool_call_start_token (str): Start token for tool calls. Default is "<function".
            tool_call_end_token (str): End token for tool calls. Default is "</function>".
            **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to ``load_dataset``.
        """

        message_transform_cls = {
            "sharegpt": ShareGPTToMessages,
            "openai": OpenAIToMessages,
        }

        if conversation_style not in message_transform_cls:
            raise ValueError(
                f"Unsupported conversation style: {conversation_style}. "
                f"Supported styles are: {list(message_transform_cls.keys())}"
            )

        message_transform = message_transform_cls[conversation_style](
            train_on_input=train_on_input,
            column_map={conversation_column: "messages"},
            new_system_prompt=new_system_prompt,
        )

        # Create the base SFT dataset
        self._ds = SFTDataset(
            source=source,
            message_transform=message_transform,
            model_transform=tokenizer,
            filter_fn=filter_fn,
            split=split,
            **load_dataset_kwargs,
        )

        # Override with enhanced transform if token classification is enabled
        if enable_token_classification:
            enhanced_transform = AgentSFTTransform(
                message_transform=message_transform,
                model_transform=tokenizer,
                enable_token_classification=True,
                train_on_tool_calls_only=train_on_tool_calls_only,   # <── pass through
                tool_call_start_token=tool_call_start_token,
                tool_call_end_token=tool_call_end_token,
            )
            self._ds._prepare_sample = enhanced_transform

        # Handle packing·
        if packed:
            self._ds = PackedDataset(self._ds, max_seq_len=tokenizer.max_seq_len)

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, index: int):
        return self._ds[index]


class AgentSFTTransform(Transform):
    """
    Enhanced SFT transform that adds token classification for agent training.
    
    This transform extends the basic SFT functionality to include:
    - Token classification for reasoning vs tool calling
    - Support for detailed metrics tracking
    """
    
    def __init__(
        self,
        message_transform: Optional[Transform] = None,
        model_transform: Optional[Transform] = None,
        enable_token_classification: bool = True,
        train_on_tool_calls_only: bool = False,
        tool_call_start_token: str = "<function",
        tool_call_end_token: str = "</function>",
    ):
        """
        Args:
            message_transform: Transform to apply to message format
            model_transform: Transform to apply for model-specific processing
            enable_token_classification: Whether to enable token classification for metrics
            train_on_tool_calls_only: Whether to replace labels of non-tool-call tokens with
                `CROSS_ENTROPY_IGNORE_IDX`. Default is False.
            tool_call_start_token: Start token for tool calls (default: "<function")
            tool_call_end_token: End token for tool calls (default: "</function>")
        """
        self._base_transform = SFTTransform(
            message_transform=message_transform,
            model_transform=model_transform,
        )
        self.enable_token_classification = enable_token_classification
        self.train_on_tool_calls_only = train_on_tool_calls_only  # <── store
        self._model_transform = model_transform
        self.tool_call_start_token = tool_call_start_token
        self.tool_call_end_token = tool_call_end_token
        
    def __call__(self, sample: Mapping[str, Any]) -> Dict[str, Any]:
        # Apply base SFT transform
        transformed_sample = self._base_transform(sample)
        
        # Add token classification if enabled
        if self.enable_token_classification and "tokens" in transformed_sample:
            try:
                # Use the model transform (tokenizer) to decode tokens
                if self._model_transform is not None:
                    # Reconstruct text from tokens for classification
                    input_ids = torch.tensor(transformed_sample["tokens"])
                    text = self._model_transform.decode(input_ids.tolist(), skip_special_tokens=False)
                    
                    # Get labels if available
                    labels = None
                    if "labels" in transformed_sample:
                        labels = torch.tensor(transformed_sample["labels"])
                    
                    # Classify tokens with configurable tokens
                    token_masks = classify_tokens_for_tool_calling(
                        text, input_ids, self._model_transform, labels,
                        self.tool_call_start_token, self.tool_call_end_token
                    )
                    # Add masks to the sample
                    transformed_sample["reasoning_mask"] = (
                        token_masks["reasoning_mask"].numpy().tolist()
                    )
                    transformed_sample["tool_call_mask"] = (
                        token_masks["tool_call_mask"].numpy().tolist()
                    )

                    if self.train_on_tool_calls_only and "labels" in transformed_sample:
                        labels[token_masks["tool_call_mask"] == 0] = CROSS_ENTROPY_IGNORE_IDX
                        transformed_sample["labels"] = labels.numpy().tolist()
            except Exception as e:
                # If token classification fails, raise the error instead of continuing
                raise RuntimeError(f"Token classification failed: {type(e).__name__}: {e}") from e
        
        return transformed_sample