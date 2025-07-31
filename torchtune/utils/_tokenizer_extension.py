# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from typing import List, Optional
from torch import nn
from torch.distributed._tensor import DTensor  # type: ignore
from torch.distributed import is_initialized, barrier, broadcast


def _resize_model_embeddings(model: nn.Module, new_vocab_size: int) -> None:
    """
    Resize embedding layers in the model to accommodate new vocabulary size.
    
    Args:
        model: The model with embedding layers to resize
        new_vocab_size: New vocabulary size
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            if module.num_embeddings < new_vocab_size:
                # Create new embedding layer with expanded size
                old_weight = module.weight.data
                new_embedding = nn.Embedding(
                    new_vocab_size, 
                    module.embedding_dim,
                    padding_idx=module.padding_idx,
                    max_norm=module.max_norm,
                    norm_type=module.norm_type,
                    scale_grad_by_freq=module.scale_grad_by_freq,
                    sparse=module.sparse,
                ).to(old_weight.device)
                
                # Copy old weights
                new_embedding.weight.data[:module.num_embeddings] = old_weight
                
                # Replace the module
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                if parent_name:
                    parent_module = model
                    for part in parent_name.split('.'):
                        parent_module = getattr(parent_module, part)
                    setattr(parent_module, child_name, new_embedding)
                else:
                    setattr(model, child_name, new_embedding)


def _init_new_token_embeddings(
    tokenizer,
    model: nn.Module,
    tokens_to_add: List[str],
    token_encodings: List[List[int]],
    original_vocab_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """
    Initialize embeddings for newly added tokens using semantic mean embedding approach.
    
    For each new token, the initialization works as follows:
    1. Use pre-computed encodings of the token string (e.g., "str_replace_editor" -> [id1, id2, id3])
    2. Get the embedding representations for those input IDs
    3. Compute the mean of these embeddings as the initialization for the new token
    
    Args:
        tokenizer: The tokenizer instance
        model: The model with embedding layers to update
        tokens_to_add: List of new tokens that were added
        token_encodings: Pre-computed encodings for each token (before they were added to tokenizer)
        original_vocab_size: Size of vocabulary before adding new tokens
        device: Device to perform computations on
        dtype: Data type for computations
    """
    if not tokens_to_add or not token_encodings:
        return
    
    # Find embedding layers in the model
    embedding_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            embedding_layers.append((name, module))
    
    if not embedding_layers:
        return
    
    with torch.no_grad():
        for i, (token, input_ids) in enumerate(zip(tokens_to_add, token_encodings)):
            new_token_id = original_vocab_size + i
            
            # Initialize embeddings for all embedding layers
            for _, embedding_layer in embedding_layers:
                if new_token_id >= embedding_layer.num_embeddings:
                    continue

                if input_ids:
                    # Convert to tensor and move to correct device
                    input_ids_tensor = torch.tensor(input_ids, device=device, dtype=torch.long)
                    
                    # Filter out any IDs that are >= original_vocab_size (avoid new tokens)
                    valid_ids = input_ids_tensor[input_ids_tensor < original_vocab_size]

                    if len(valid_ids) > 0:
                        # Get token embeddings for semantic initialization
                        weight = embedding_layer.weight
                        if isinstance(weight, DTensor):
                            # For DTensor, gather the full tensor from all ranks to access all embeddings
                            # full_tensor() performs necessary collectives to gather local tensors from all ranks
                            weight = weight.full_tensor()

                        token_embeddings = weight[valid_ids]
                        
                        mean_embedding = token_embeddings.mean(dim=0).to(dtype)
                        _sync_parameter_update(embedding_layer, new_token_id, mean_embedding)
                        continue

                # Fallback to random sampling if no valid encoding
                sample_size = min(100, original_vocab_size)
                sample_ids = torch.randint(0, original_vocab_size, (sample_size,), device=device, dtype=torch.long)

                # Get sample embeddings robustly (DTensor aware)
                sample_weight = embedding_layer.weight
                if isinstance(sample_weight, DTensor):
                    sample_weight = sample_weight.full_tensor()
                sample_embeddings = sample_weight[sample_ids]
                mean_embedding = sample_embeddings.mean(dim=0).to(dtype)
                _sync_parameter_update(embedding_layer, new_token_id, mean_embedding)


def _is_distributed_training():
    """
    Check if we're actually in a distributed training context.
    Returns True only if distributed is initialized and world_size > 1.
    """
    return (torch.distributed.is_available() and 
            torch.distributed.is_initialized() and 
            torch.distributed.get_world_size() > 1)


def _sync_parameter_update(
    embedding_layer: nn.Embedding,
    new_token_id: int,
    mean_embedding: torch.Tensor,
):
    """
    Update embedding parameter and ensure synchronization in distributed training.

    Args:
        embedding_layer: The embedding layer to update.
        new_token_id: Index of the new token (global index).
        mean_embedding: A 1D tensor of size [embedding_dim].
    """
    weight = embedding_layer.weight

    # 1) DTensor (sharded) path: modify only the local slice
    if isinstance(weight, DTensor):
        from torch.distributed._tensor import Shard
        
        # Check if the tensor is row-sharded
        is_row_sharded = any(isinstance(p, Shard) and p.dim == 0 for p in weight.placements)
        
        if is_row_sharded:
            # For row-sharded tensors, we need to determine which rank owns this row
            global_shape = weight.shape
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            
            # Calculate the range of rows owned by each rank
            rows_per_rank = global_shape[0] // world_size
            remainder = global_shape[0] % world_size
            
            # Ranks 0 to remainder-1 get rows_per_rank + 1 rows
            # Ranks remainder to world_size-1 get rows_per_rank rows
            if rank < remainder:
                start_row = rank * (rows_per_rank + 1)
                end_row = start_row + rows_per_rank + 1
            else:
                start_row = remainder * (rows_per_rank + 1) + (rank - remainder) * rows_per_rank
                end_row = start_row + rows_per_rank
            
            # Check if this rank owns the row for new_token_id
            if start_row <= new_token_id < end_row:
                # Convert global index to local index
                local_index = new_token_id - start_row
                local_shard = weight.to_local()
                local_shard[local_index].copy_(mean_embedding)
            
            # Synchronize across all ranks (just a barrier, no broadcast needed)
            if is_initialized():
                torch.distributed.barrier()
        else:
            # For replicated tensors, all ranks have the full tensor
            local_shard = weight.to_local()
            local_shard[new_token_id].copy_(mean_embedding)
            
            # No broadcast needed - each rank updates its own copy
            if is_initialized():
                torch.distributed.barrier()

        return

    # 2) Non-DTensor, non-distributed: single-GPU
    if not (is_initialized() and torch.distributed.get_world_size() > 1):
        weight.data[new_token_id].copy_(mean_embedding)
        return

    # 3) Vanilla distributed (e.g. DDP or ZeRO without DTensor):
    rank = torch.distributed.get_rank()
    if rank == 0:
        weight.data[new_token_id].copy_(mean_embedding)

    # Ensure all ranks wait
    torch.distributed.barrier()

    # Broadcast the updated parameter to all ranks
    torch.distributed.broadcast(weight.data, src=0)


def extend_tokenizer_if_needed(
    tokenizer,
    tool_call_special_tokens: List[str],
    model: nn.Module,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[List[str]]:
    """
    Extend tokenizer with tool call special tokens if needed.
    
    Args:
        tokenizer: The tokenizer instance
        tool_call_special_tokens: List of special tokens to add
        model: The model instance
        device: Device for computations
        dtype: Data type for computations
    
    Returns:
        List of tokens that were added, or None if no tokens were added
    """
    if not tool_call_special_tokens:
        return None
    
    # Find tokens that need to be added
    existing_vocab = tokenizer.encoder
    tokens_to_add = [tok for tok in tool_call_special_tokens if tok not in existing_vocab]
    
    if not tokens_to_add:
        return None
    
    # Pre-compute encodings for semantic initialization (before adding tokens to tokenizer)
    original_vocab_size = len(existing_vocab)
    token_encodings = []
    
    for token in tokens_to_add:
        try:
            input_ids = tokenizer.encode(token, add_bos=False, add_eos=False)
            token_encodings.append(input_ids)
        except Exception as e:
            print(f"Warning: Failed to encode token '{token}' for semantic initialization: {e}")
            token_encodings.append([])  # Empty list as fallback
    
    # Find the next available token ID by checking existing tokens
    # We need to find the maximum token ID across all tokenizer dictionaries
    max_existing_id = 0
    
    # Check all possible sources of existing token IDs
    if hasattr(tokenizer, 'special_tokens') and tokenizer.special_tokens:
        max_existing_id = max(max_existing_id, max(tokenizer.special_tokens.values()))
    
    if hasattr(tokenizer, 'encoder') and tokenizer.encoder:
        max_existing_id = max(max_existing_id, max(tokenizer.encoder.values()))
    
    # Fallback to original_vocab_size if no existing tokens found
    if max_existing_id == 0:
        max_existing_id = original_vocab_size - 1
    
    # Start assigning new token IDs from the next available ID
    next_available_id = max_existing_id + 1
    
    # Manually add new tokens to tokenizer
    num_added = 0
    
    for token in tokens_to_add:
        # Add to special tokens with new id
        new_token_id = next_available_id + num_added
        tokenizer.special_tokens[token] = new_token_id
        tokenizer._special_tokens_reversed[new_token_id] = token
        # Also add to encoder for consistency
        tokenizer.encoder[token] = new_token_id
        tokenizer.decoder[new_token_id] = token
        num_added += 1
    
    if num_added > 0:
        # Update the pattern for special tokens
        import regex as re
        tokenizer._pattern_split_special_tokens = re.compile(
            r"(\L<options>)", options=tokenizer.special_tokens.keys()
        )
        
        # Resize model embeddings to accommodate new tokens
        new_total_vocab_size = next_available_id + num_added
        _resize_model_embeddings(model, new_total_vocab_size)
        
        # Initialize embeddings for the new tokens using pre-computed encodings
        _init_new_token_embeddings(tokenizer, model, tokens_to_add, token_encodings, original_vocab_size, device, dtype)
        
        print(f"Added {num_added} special tokens to tokenizer: {tokens_to_add}")
        return tokens_to_add
    
    return None