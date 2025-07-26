# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional
from torchtune.data._common import CROSS_ENTROPY_IGNORE_IDX

class DetailedMetricsTracker:
    """
    Tracks detailed metrics during training including entropy, perplexity, 
    and top-k probabilities for different token types (reasoning vs tool calling).
    
    This class is designed for agent SFT training where we want to monitor
    model performance on different types of tokens separately.
    """
    
    def __init__(
        self,
        entropy_k: int = 10,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Args:
            entropy_k: Number of top probabilities to compute for entropy calculation
            device: Device to run computations on
            dtype: Data type for computations
        """
        self.entropy_k = entropy_k
        self.device = device or torch.device("cpu")
        self.dtype = dtype or torch.float32
        
    def compute_metrics(
        self,
        logits: torch.Tensor,  # (batch_size, seq_len, vocab_size)
        labels: torch.Tensor,  # (batch_size, seq_len)
        loss: torch.Tensor,    # scalar loss
        reasoning_mask: Optional[torch.Tensor] = None,  # (batch_size, seq_len)
        tool_call_mask: Optional[torch.Tensor] = None,  # (batch_size, seq_len)
    ) -> Dict[str, float]:
        """
        Compute detailed metrics for the given logits, labels, and masks.
        
        Args:
            logits: Model logits
            labels: Target labels
            loss: Computed loss value
            reasoning_mask: Mask indicating reasoning tokens
            tool_call_mask: Mask indicating tool call tokens
            
        Returns:
            Dictionary of computed metrics
        """
        with torch.no_grad():
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Valid positions mask
            valid_mask = shift_labels != CROSS_ENTROPY_IGNORE_IDX
            if valid_mask.sum() == 0:
                return {}
            
            # Flatten and extract valid positions
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_labels = shift_labels.view(-1)
            flat_valid_mask = valid_mask.view(-1)
            
            valid_logits = flat_logits[flat_valid_mask]
            valid_labels = flat_labels[flat_valid_mask]
            
            # Compute probabilities once with numerical stability
            # Clamp logits to prevent overflow in softmax
            stable_logits = torch.clamp(valid_logits, min=-50.0, max=50.0)
            probs = torch.softmax(stable_logits, dim=-1)
            k = min(self.entropy_k, valid_logits.size(-1))
            top_k_probs, _ = torch.topk(probs, k=k, dim=-1)
            
            # Ensure probabilities are valid
            top_k_probs = torch.clamp(top_k_probs, min=1e-10, max=1.0)
            
            metrics = {}
            
            # Overall metrics
            metrics['entropy'] = self._compute_entropy(top_k_probs[:, 0])
            metrics['perplexity'] = self._compute_perplexity(loss)
            
            # Top-k probabilities with safety checks
            for top_n in [1, 2, 5, 10]:
                if top_n <= k:
                    top_n_sum = top_k_probs[:, :top_n].sum(dim=-1)
                    # Ensure finite values
                    top_n_sum = torch.where(torch.isfinite(top_n_sum), top_n_sum, torch.zeros_like(top_n_sum))
                    mean_val = top_n_sum.mean().item()
                    metrics[f'top{top_n}_prob'] = mean_val if not torch.isnan(torch.tensor(mean_val)) else 0.0
            
            # Masked metrics for reasoning and tool_call
            if reasoning_mask is not None or tool_call_mask is not None:
                masked_metrics = self._compute_masked_metrics(
                    valid_mask, flat_valid_mask, top_k_probs, loss, k,
                    reasoning_mask, tool_call_mask
                )
                metrics.update(masked_metrics)
            
            return metrics
    
    def _compute_entropy(self, top1_probs: torch.Tensor) -> float:
        """Compute binary Shannon entropy efficiently."""
        if top1_probs.numel() == 0:
            return 0.0
            
        epsilon = 1e-8
        p = torch.clamp(top1_probs, min=epsilon, max=1.0 - epsilon)
        
        # Compute log terms safely
        log_p = torch.log2(p)
        log_1_minus_p = torch.log2(1 - p)
        
        # Check for NaN or inf values and replace with 0
        log_p = torch.where(torch.isfinite(log_p), log_p, torch.zeros_like(log_p))
        log_1_minus_p = torch.where(torch.isfinite(log_1_minus_p), log_1_minus_p, torch.zeros_like(log_1_minus_p))
        
        entropy = -(p * log_p + (1 - p) * log_1_minus_p)
        
        # Final safety check for NaN values
        entropy = torch.where(torch.isfinite(entropy), entropy, torch.zeros_like(entropy))
        
        result = entropy.mean().item()
        return result if not torch.isnan(torch.tensor(result)) else 0.0
    
    def _compute_perplexity(self, loss: torch.Tensor) -> float:
        """Compute perplexity with numerical stability."""
        # Clamp loss to prevent overflow in exp
        clamped_loss = torch.clamp(loss, min=-50.0, max=50.0)
        perplexity = torch.exp(clamped_loss).item()
        
        # Cap perplexity at reasonable maximum
        return min(perplexity, 1e6) if not torch.isnan(torch.tensor(perplexity)) else 1e6
    
    def _compute_masked_metrics(
        self,
        valid_mask: torch.Tensor,  # (batch_size, seq_len-1)
        flat_valid_mask: torch.Tensor,  # (batch_size*(seq_len-1),)
        all_top_k_probs: torch.Tensor,  # (num_valid, k)
        loss: torch.Tensor,
        k: int,
        reasoning_mask: Optional[torch.Tensor] = None,
        tool_call_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Compute metrics for reasoning and tool_call tokens."""
        metrics = {}
        
        if reasoning_mask is not None:
            # Get shifted mask to align with labels
            reasoning_mask_shifted = reasoning_mask[:, :-1].to(valid_mask.device)
            
            # Combine with valid mask and flatten
            reasoning_valid = (reasoning_mask_shifted & valid_mask).view(-1)
            
            # Map to valid_logits indices
            valid_indices = torch.cumsum(flat_valid_mask, dim=0) - 1
            reasoning_indices = valid_indices[reasoning_valid]
            
            if reasoning_indices.numel() > 0 and reasoning_indices.max() < all_top_k_probs.size(0):
                reasoning_probs = all_top_k_probs[reasoning_indices]
                
                metrics['reasoning_entropy'] = self._compute_entropy(reasoning_probs[:, 0])
                metrics['reasoning_perplexity'] = self._compute_perplexity(loss)
                
                for top_n in [1, 2, 5, 10]:
                    if top_n <= k:
                        top_n_sum = reasoning_probs[:, :top_n].sum(dim=-1)
                        top_n_sum = torch.where(torch.isfinite(top_n_sum), top_n_sum, torch.zeros_like(top_n_sum))
                        mean_val = top_n_sum.mean().item()
                        metrics[f'reasoning_top{top_n}_prob'] = mean_val if not torch.isnan(torch.tensor(mean_val)) else 0.0
        
        if tool_call_mask is not None:
            # Get shifted mask to align with labels
            tool_call_mask_shifted = tool_call_mask[:, :-1].to(valid_mask.device)
            
            # Combine with valid mask and flatten
            tool_call_valid = (tool_call_mask_shifted & valid_mask).view(-1)
            
            # Map to valid_logits indices
            valid_indices = torch.cumsum(flat_valid_mask, dim=0) - 1
            tool_call_indices = valid_indices[tool_call_valid]
            
            if tool_call_indices.numel() > 0 and tool_call_indices.max() < all_top_k_probs.size(0):
                tool_call_probs = all_top_k_probs[tool_call_indices]
                
                metrics['tool_call_entropy'] = self._compute_entropy(tool_call_probs[:, 0])
                metrics['tool_call_perplexity'] = self._compute_perplexity(loss)
                
                for top_n in [1, 2, 5, 10]:
                    if top_n <= k:
                        top_n_sum = tool_call_probs[:, :top_n].sum(dim=-1)
                        top_n_sum = torch.where(torch.isfinite(top_n_sum), top_n_sum, torch.zeros_like(top_n_sum))
                        mean_val = top_n_sum.mean().item()
                        metrics[f'tool_call_top{top_n}_prob'] = mean_val if not torch.isnan(torch.tensor(mean_val)) else 0.0
        
        return metrics