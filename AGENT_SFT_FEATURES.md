# Agent SFT Training with Detailed Metrics

This document describes the new features implemented for agent SFT training in torchtune, including detailed metrics tracking and token classification for reasoning vs tool calling.

## Overview

The new agent SFT features include:

1. **Detailed Metrics Tracking**: Track entropy, perplexity, and top-k probabilities during training and validation
2. **Token Classification**: Classify tokens as either reasoning or tool calling for specialized metrics
3. **Enhanced Data Processing**: Automatic detection and classification of `<function>...</function>` blocks
4. **New Training Recipe**: Dedicated training recipe for agent SFT with integrated metrics

## Key Components

### 1. DetailedMetricsTracker

Located in `torchtune/training/_detailed_metrics.py`

This class computes detailed metrics including:
- **Entropy**: Binary Shannon entropy from top-1 probabilities
- **Perplexity**: Exponential of the loss, capped at 1e6
- **Top-k Probabilities**: Accumulated probabilities for top 1, 2, 5, and 10 tokens
- **Masked Metrics**: All above metrics computed separately for reasoning and tool calling tokens

### 2. AgentSFTDataset and AgentSFTTransform

Located in `torchtune/datasets/_agent_sft.py`

These classes extend the standard SFT functionality with:
- **Token Classification**: Automatic classification of tokens into reasoning vs tool calling
- **Function Block Detection**: Uses regex to identify `<function>...</function>` blocks
- **Enhanced Preprocessing**: Maintains compatibility with existing SFT while adding agent-specific features

### 3. Agent SFT Training Recipe

Located in `recipes/agent_sft_distributed.py`

A new training recipe that:
- Integrates detailed metrics tracking into the training loop
- Supports all features of the standard distributed training recipe
- Adds agent-specific configurations for metrics tracking
- Logs detailed metrics during both training and validation

## Usage

### 1. Basic Configuration

The provided configuration template at `recipes/configs/agent_sft_distributed.yaml` shows how to use AgentSFTDataset with the standard tokenizer pattern:

```yaml
# Model and tokenizer (standard torchtune pattern)
model:
  _component_: torchtune.models.qwen2_5.qwen2_5_32b_instruct

tokenizer:
  _component_: torchtune.models.qwen2_5.qwen2_5_tokenizer
  path: llm-weights/SWE-bench/SWE-agent-LM-32B/vocab.json
  merges_file: llm-weights/SWE-bench/SWE-agent-LM-32B/merges.txt
  max_seq_len: 32768

# Dataset using AgentSFTDataset (receives tokenizer from recipe setup)
dataset:
  _component_: torchtune.datasets.AgentSFTDataset
  source: json
  data_files: datasets/your_data.jsonl
  message_transform:
    _component_: torchtune.data.JSONToMessages
    column_map:
      messages: messages
  enable_token_classification: True  # Enable agent-specific features

# Agent SFT specific configurations
enable_detailed_metrics: True
enable_token_classification: True
entropy_k: 10
log_detailed_metrics_every_n_steps: 10
```

### 2. Running Training

```bash
# Multi-GPU training (example with 2 GPUs)
tune run --nnodes 1 --nproc_per_node 2 agent_sft_distributed --config agent_sft_distributed

# With custom overrides
tune run --nnodes 1 --nproc_per_node 2 agent_sft_distributed \
    --config agent_sft_distributed \
    checkpointer.checkpoint_dir=/path/to/model \
    dataset.data_files=datasets/my_data.jsonl
```

### 3. Expected Metrics

The training will log the following metrics:

**Overall Metrics:**
- `train_entropy` / `val_entropy`: Overall sequence entropy
- `train_perplexity` / `val_perplexity`: Overall perplexity
- `train_top1_prob` / `val_top1_prob`: Top-1 token probability
- `train_top2_prob` / `val_top2_prob`: Top-2 token probabilities
- `train_top5_prob` / `val_top5_prob`: Top-5 token probabilities
- `train_top10_prob` / `val_top10_prob`: Top-10 token probabilities

**Reasoning-Specific Metrics:**
- `train_reasoning_entropy` / `val_reasoning_entropy`: Entropy for reasoning tokens
- `train_reasoning_perplexity` / `val_reasoning_perplexity`: Perplexity for reasoning tokens
- `train_reasoning_top*_prob` / `val_reasoning_top*_prob`: Top-k probabilities for reasoning tokens

**Tool Call-Specific Metrics:**
- `train_tool_call_entropy` / `val_tool_call_entropy`: Entropy for tool call tokens
- `train_tool_call_perplexity` / `val_tool_call_perplexity`: Perplexity for tool call tokens
- `train_tool_call_top*_prob` / `val_tool_call_top*_prob`: Top-k probabilities for tool call tokens

## Token Classification

The token classification works by:

1. **Function Block Detection**: Uses regex pattern `<function=[^>]+>.*?</function>` to identify tool call blocks
2. **Token Mapping**: Maps each token to character positions in the original text
3. **Classification**: Tokens overlapping with function blocks are classified as `TOOL_CALL`, others as `REASONING`
4. **Special Token Handling**: Special tokens (pad, eos, bos) are excluded from all metrics

## Configuration Options

### Agent SFT Specific Options

- `enable_detailed_metrics` (bool): Enable detailed metrics tracking (default: True)
- `enable_token_classification` (bool): Enable token classification (default: True)
- `entropy_k` (int): Number of top probabilities for entropy calculation (default: 10)
- `log_detailed_metrics_every_n_steps` (int): Log detailed metrics every N steps (default: 10)

### Dataset Options

- `enable_token_classification` (bool): Enable token classification in dataset preprocessing

## Compatibility

The new features are designed to be fully compatible with existing torchtune functionality:

- **Backward Compatibility**: Regular SFT datasets work without modification
- **Fallback Support**: If token classification is disabled or fails, training continues with standard metrics
- **Minimal Changes**: Existing training recipes continue to work unchanged

## Performance Considerations

- **Memory**: Detailed metrics computation adds minimal memory overhead
- **Compute**: Token classification adds small preprocessing overhead
- **Logging**: Detailed metrics are logged only at specified intervals to minimize I/O impact

## Integration with Existing Code

The new features integrate seamlessly with torchtune's existing systems:

- **Config System**: Uses torchtune's config instantiation
- **Distributed Training**: Fully compatible with FSDP and tensor parallelism
- **Checkpointing**: Works with all existing checkpointing strategies
- **Metric Logging**: Integrates with WandB, TensorBoard, and other loggers