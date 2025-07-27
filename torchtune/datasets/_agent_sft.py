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
    
    Simplified two-level classification:
    - REASONING: Normal conversation text outside <function>...</function> blocks
    - TOOL_CALL: Entire <function>...</function> blocks as holistic units
    
    Args:
        text: Original text string
        input_ids: Tokenized input IDs
        tokenizer: Tokenizer used for encoding
        
    Returns:
        Dictionary containing reasoning_mask and tool_call_mask tensors
    """
    # Find all function blocks (entire <function>...</function> sections)
    function_block_pattern = re.compile(r'<function=[^>]+>.*?</function>', re.DOTALL)
    function_blocks = list(function_block_pattern.finditer(text))
    
    seq_len = len(input_ids)
    
    # Initialize classification masks
    reasoning_mask = torch.ones(seq_len, dtype=torch.bool)  # Default to reasoning
    tool_call_mask = torch.zeros(seq_len, dtype=torch.bool)
    
    # Get special token IDs efficiently
    def get_special_token_ids(tokenizer):
        """Get special token IDs using tokenizer's native methods."""
        special_ids = set()
        
        # TorchTune tokenizer: use special_tokens dict
        if hasattr(tokenizer, 'special_tokens') and isinstance(tokenizer.special_tokens, dict):
            special_ids.update(tokenizer.special_tokens.values())
        
        # HuggingFace tokenizer: use all_special_ids
        elif hasattr(tokenizer, 'all_special_ids'):
            special_ids.update(tokenizer.all_special_ids)
        
        return special_ids
    
    special_token_ids = get_special_token_ids(tokenizer)
    
    # Get token text using the best available method
    def get_token_text(tokenizer, token_id):
        """Get text representation of a token."""
        # Method 1: TorchTune style - use _convert_id_to_token + _convert_tokens_to_string
        if hasattr(tokenizer, '_convert_id_to_token') and hasattr(tokenizer, '_convert_tokens_to_string'):
            token = tokenizer._convert_id_to_token(token_id)
            return tokenizer._convert_tokens_to_string([token])
        
        # Method 2: HF style - use convert_ids_to_tokens + convert_tokens_to_string  
        elif hasattr(tokenizer, 'convert_ids_to_tokens') and hasattr(tokenizer, 'convert_tokens_to_string'):
            token = tokenizer.convert_ids_to_tokens([token_id])[0]
            return tokenizer.convert_tokens_to_string([token])
        
        # Method 3: Universal fallback - decode single token
        else:
            try:
                return tokenizer.decode([token_id], skip_special_tokens=False)
            except Exception:
                return ""
    
    # Build character spans for each token
    token_char_spans = []
    cumulative_pos = 0
    
    for i, token_id in enumerate(input_ids.tolist()):
        # Skip special tokens - they don't contribute to reasoning/tool_call classification
        if token_id in special_token_ids:
            reasoning_mask[i] = False
            token_char_spans.append((cumulative_pos, cumulative_pos))
            continue
        
        # Get token text
        token_text = get_token_text(tokenizer, token_id)
        
        # Skip empty tokens
        if not token_text:
            reasoning_mask[i] = False
            token_char_spans.append((cumulative_pos, cumulative_pos))
            continue
        
        # Record character span for this token
        start_pos = cumulative_pos
        end_pos = cumulative_pos + len(token_text)
        cumulative_pos = end_pos
        token_char_spans.append((start_pos, end_pos))
    
    # Classify tokens based on their position relative to function blocks
    for i, (start_pos, end_pos) in enumerate(token_char_spans):
        if reasoning_mask[i] == False:  # Skip special tokens
            continue
            
        # Check if token overlaps with any function block
        is_in_tool_call = False
        overlapping_block = None
        for block_idx, block in enumerate(function_blocks):
            block_start, block_end = block.span()
            
            # Check if token overlaps with this function block
            if not (end_pos <= block_start or start_pos >= block_end):
                is_in_tool_call = True
                overlapping_block = block_idx
                break
        
        if is_in_tool_call:
            # Token is inside a function block -> TOOL_CALL
            reasoning_mask[i] = False
            tool_call_mask[i] = True
        # else: Token is outside function blocks -> REASONING (already True by default)
    
    return {
        "reasoning_mask": reasoning_mask,  
        "tool_call_mask": tool_call_mask
    }


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
                # If token classification fails, continue without it
                # This ensures compatibility when tokenizer doesn't support the required methods
                print(f"⚠️  Token classification failed: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
        
        return transformed_sample