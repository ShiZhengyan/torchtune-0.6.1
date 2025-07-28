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
    tokenizer
) -> Dict[str, torch.Tensor]:
    """
    Classify tokens for tool calling metrics during preprocessing.
    
    Simplified approach that avoids complex character-to-token mapping:
    - REASONING: Normal conversation text outside <function>...</function> blocks
    - TOOL_CALL: Entire <function>...</function> blocks as holistic units
    
    Args:
        text: Original text string
        input_ids: Tokenized input IDs
        tokenizer: Tokenizer used for encoding
        
    Returns:
        Dictionary containing reasoning_mask and tool_call_mask tensors
    """
    seq_len = len(input_ids)
    
    # Initialize classification masks - default everything to reasoning
    reasoning_mask = torch.ones(seq_len, dtype=torch.bool)
    tool_call_mask = torch.zeros(seq_len, dtype=torch.bool)
    
    try:
        # Find all function blocks in the original text
        function_block_pattern = re.compile(r'<function=[^>]+>.*?</function>', re.DOTALL)
        function_blocks = list(function_block_pattern.finditer(text))
        
        # If no function blocks found, everything is reasoning
        if not function_blocks:
            return {
                "reasoning_mask": reasoning_mask,  
                "tool_call_mask": tool_call_mask
            }
        
        # Get special token IDs to exclude from classification
        special_token_ids = get_special_token_ids(tokenizer)
        
        # Simple approach: estimate which portion of tokens correspond to function blocks
        # based on character proportions in the original text
        total_text_chars = len(text)
        total_function_chars = sum(block.end() - block.start() for block in function_blocks)
        
        if total_text_chars > 0 and total_function_chars > 0:
            # Rough estimate of what fraction of tokens should be tool_call
            function_ratio = total_function_chars / total_text_chars
            estimated_tool_tokens = int(function_ratio * seq_len)
            
            # Find sequences that look like function calls by pattern matching
            # Look for patterns that might indicate function boundaries
            tool_call_regions = find_tool_call_token_regions(
                input_ids, tokenizer, special_token_ids, estimated_tool_tokens
            )
            
            # Mark identified regions as tool_call tokens
            for start_idx, end_idx in tool_call_regions:
                for i in range(start_idx, min(end_idx, seq_len)):
                    if input_ids[i].item() not in special_token_ids:
                        reasoning_mask[i] = False
                        tool_call_mask[i] = True
        
        # Mark special tokens as neither reasoning nor tool_call
        for i, token_id in enumerate(input_ids.tolist()):
            if token_id in special_token_ids:
                reasoning_mask[i] = False
                tool_call_mask[i] = False
        
        return {
            "reasoning_mask": reasoning_mask,  
            "tool_call_mask": tool_call_mask
        }
        
    except Exception as e:
        # If classification fails completely, raise the error instead of silently continuing
        raise RuntimeError(f"Token classification failed: {type(e).__name__}: {e}") from e


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


def find_tool_call_token_regions(input_ids, tokenizer, special_token_ids, estimated_tool_tokens):
    """
    Find token regions that likely correspond to tool calls using simple heuristics.
    
    This is a simplified approach that works reasonably well without complex text reconstruction.
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
                # Look for function start patterns
                for pattern in ['<function']:
                    try:
                        tokens = tokenizer.encode(pattern, add_bos=False, add_eos=False)
                        if tokens:
                            function_start_tokens.extend(tokens)
                    except Exception as encode_error:
                        raise RuntimeError(f"Failed to encode function start pattern '{pattern}': {type(encode_error).__name__}: {encode_error}") from encode_error
                
                # Look for function end patterns
                for pattern in ['</function>']:
                    try:
                        tokens = tokenizer.encode(pattern, add_bos=False, add_eos=False)
                        if tokens:
                            function_end_tokens.extend(tokens)
                    except Exception as encode_error:
                        raise RuntimeError(f"Failed to encode function end pattern '{pattern}': {type(encode_error).__name__}: {encode_error}") from encode_error
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
            )
            self._ds._prepare_sample = enhanced_transform

        # Handle packing
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
    ):
        """
        Args:
            message_transform: Transform to apply to message format
            model_transform: Transform to apply for model-specific processing
            enable_token_classification: Whether to enable token classification for metrics
        """
        self._base_transform = SFTTransform(
            message_transform=message_transform,
            model_transform=model_transform,
        )
        self.enable_token_classification = enable_token_classification
        self._model_transform = model_transform
        
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
                    
                    # Classify tokens
                    token_masks = classify_tokens_for_tool_calling(text, input_ids, self._model_transform)
                    
                    # Add masks to the sample
                    transformed_sample["reasoning_mask"] = token_masks["reasoning_mask"].numpy().tolist()
                    transformed_sample["tool_call_mask"] = token_masks["tool_call_mask"].numpy().tolist()
            except Exception as e:
                # If token classification fails, raise the error instead of continuing
                raise RuntimeError(f"Token classification failed: {type(e).__name__}: {e}") from e
        
        return transformed_sample