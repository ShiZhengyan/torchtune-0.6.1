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
    
    # Convert input_ids to tokens for classification
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    seq_len = len(tokens)
    
    # Initialize classification masks
    reasoning_mask = torch.ones(seq_len, dtype=torch.bool)  # Default to reasoning
    tool_call_mask = torch.zeros(seq_len, dtype=torch.bool)
    
    # Reconstruct text with token-to-position mapping
    token_char_spans = []
    reconstructed_text = ""
    
    for i, token in enumerate(tokens):
        # Handle special tokens
        if hasattr(tokenizer, 'pad_token') and token == tokenizer.pad_token:
            reasoning_mask[i] = False
            token_char_spans.append((len(reconstructed_text), len(reconstructed_text)))
            continue
        if hasattr(tokenizer, 'eos_token') and token == tokenizer.eos_token:
            reasoning_mask[i] = False
            token_char_spans.append((len(reconstructed_text), len(reconstructed_text)))
            continue
        if hasattr(tokenizer, 'bos_token') and token == tokenizer.bos_token:
            reasoning_mask[i] = False
            token_char_spans.append((len(reconstructed_text), len(reconstructed_text)))
            continue
        
        # Convert token to text
        token_text = tokenizer.convert_tokens_to_string([token])
        
        # Record the character span for this token
        start_pos = len(reconstructed_text)
        reconstructed_text += token_text
        end_pos = len(reconstructed_text)
        token_char_spans.append((start_pos, end_pos))
    
    # Classify tokens based on their position relative to function blocks
    for i, (start_pos, end_pos) in enumerate(token_char_spans):
        if reasoning_mask[i] == False:  # Skip special tokens
            continue
            
        # Check if token overlaps with any function block
        is_in_tool_call = False
        for block in function_blocks:
            block_start, block_end = block.span()
            
            # Check if token overlaps with this function block
            if not (end_pos <= block_start or start_pos >= block_end):
                is_in_tool_call = True
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
                pass
        
        return transformed_sample