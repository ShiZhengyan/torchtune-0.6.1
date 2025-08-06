# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
import time

from functools import partial
from typing import Any, Dict, List, Optional, Union
from warnings import warn

import torch
from omegaconf import DictConfig, ListConfig

from torch import nn
from torch.distributed import (
    destroy_process_group,
    init_device_mesh,
    init_process_group,
)
from torch.distributed._tensor import DTensor
from torch.distributed.tensor.parallel import parallelize_module
from torch.optim import Optimizer
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler
from torchtune import config, modules, training, utils
from torchtune.config._utils import _get_component_from_path
from torchtune.data import padded_collate_packed
from torchtune.datasets import ConcatDataset
from torchtune.datasets._agent_sft import AgentSFTDataset
from torchtune.recipe_interfaces import FTRecipeInterface
from torchtune.training import DummyProfiler, PROFILER_KEY
from torchtune.training._detailed_metrics import DetailedMetricsTracker
from torchtune.training.activations import apply_selective_activation_checkpointing
from torchtune.training.checkpointing._checkpoint_client import (
    CheckpointClient,
    TrainingProgress,
)
from torchtune.training.lr_schedulers import get_lr

from tqdm import tqdm

log = utils.get_logger("DEBUG")


class AgentSFTRecipeDistributed(FTRecipeInterface):
    """
    Agent SFT training recipe with detailed metrics tracking for distributed training.
    
    This recipe extends the base full finetuning functionality with:
    - Detailed metrics tracking (entropy, perplexity, top-k probabilities)
    - Token classification for reasoning vs tool calling
    - Enhanced data preprocessing for agent training
    
    All features from FullFinetuneRecipeDistributed are supported, plus agent-specific enhancements.
    """

    def __init__(self, cfg: DictConfig) -> None:
        device_type = cfg.device
        self._device = utils.get_device(device=device_type)
        self._dtype = training.get_dtype(cfg.dtype, device=self._device)

        if self._dtype == torch.float16:
            raise ValueError(
                "full fp16 training is not supported with this recipe. Please use bf16 or fp32 instead."
            )

        # Set up the backend for distributed training (NCCL, GLOO, etc.)
        self._enable_async_checkpointing = cfg.get("enable_async_checkpointing", False)
        self.fsdp_cpu_offload = cfg.get("fsdp_cpu_offload", False)
        self.distributed_backend = training.get_distributed_backend(
            device_type,
            offload_ops_to_cpu=self.fsdp_cpu_offload
            or self._enable_async_checkpointing,
        )
        init_process_group(self.distributed_backend)

        # Initialize distributed variables
        self.world_size, self.rank = utils.get_world_size_and_rank()
        self._is_rank_zero = self.rank == 0
        self.tensor_parallel_plan = config.instantiate(
            cfg.get("tensor_parallel_plan", None)
        )
        self.tensor_parallel_dim = cfg.get("tensor_parallel_dim", 1)
        if self.tensor_parallel_dim > 1 and self.tensor_parallel_plan is None:
            raise ValueError(
                "Tensor Parallel plan needs to be provided when tensor parallel is enabled."
            )
        if self.world_size % self.tensor_parallel_dim != 0:
            raise ValueError(
                f"world_size {self.world_size} must be divisible by tensor_parallel_dim {self.tensor_parallel_dim}"
            )
        if self.tensor_parallel_dim > 1 and cfg.optimizer.get("fused", False):
            raise ValueError(
                "Tensor parallelism is currently incompatible with fused optimizer."
            )

        self.data_parallel_dim = self.world_size // self.tensor_parallel_dim

        # Logging attributes
        self._output_dir = cfg.output_dir
        self._log_every_n_steps = cfg.get("log_every_n_steps", 1)
        self._log_peak_memory_stats = cfg.get("log_peak_memory_stats", False)
        if self._log_peak_memory_stats and device_type != "cuda":
            log.info(
                "log_peak_memory_stats was set to True, however, training does not use cuda. Setting log_peak_memory_stats=False."
            )
            self._log_peak_memory_stats = False

        # Training cfg
        self._resume_from_checkpoint = cfg.resume_from_checkpoint
        self._gradient_accumulation_steps = cfg.gradient_accumulation_steps
        self._optimizer_in_bwd = cfg.get("optimizer_in_bwd", False)
        self._clip_grad_norm = cfg.get("clip_grad_norm", None)
        self._checkpoint_client = CheckpointClient(cfg)
        
        # Validation cfg
        self._enable_validation = cfg.get("dataset_val", None) is not None
        self._run_val_every_n_steps = cfg.get("run_val_every_n_steps", 500) if self._enable_validation else None
        self._batch_size_val = cfg.get("batch_size_val", cfg.batch_size) if self._enable_validation else None
        self._last_val_loss = None  # Track the latest validation loss for display

        # Agent SFT specific configurations
        self._enable_detailed_metrics = cfg.get("enable_detailed_metrics", True)
        self._entropy_k = cfg.get("entropy_k", 10)
        self._log_detailed_metrics_every_n_steps = cfg.get("log_detailed_metrics_every_n_steps", 10)

        # Initialize detailed metrics tracker
        if self._enable_detailed_metrics:
            self._metrics_tracker = DetailedMetricsTracker(
                entropy_k=self._entropy_k,
                device=self._device,
                dtype=self._dtype,
            )
            if self._is_rank_zero:
                log.info(f"📊 Detailed metrics tracking enabled with entropy_k={self._entropy_k}")

        # Optimizer in backward is not compatible with gradient accumulation or gradient clipping
        if self._optimizer_in_bwd:
            if self._clip_grad_norm is not None:
                raise RuntimeError(
                    "Gradient clipping is not supported with optimizer in bwd."
                    "Please set clip_grad_norm=None, or optimizer_in_bwd=False."
                )
            if self._gradient_accumulation_steps > 1:
                raise RuntimeError(
                    "Gradient accumulation is not supported with optimizer in bwd."
                    "Please set gradient_accumulation_steps=1, or optimizer_in_bwd=False."
                )

        # activation checkpointing/offloading
        self._enable_activation_checkpointing = cfg.get(
            "enable_activation_checkpointing", False
        )
        self._enable_activation_offloading = cfg.get(
            "enable_activation_offloading", False
        )
        if self._enable_activation_offloading:
            if device_type != "cuda":
                raise RuntimeError(
                    "enable_activation_offloading should only be True when training on CUDA"
                )
            if not self._enable_activation_checkpointing:
                raise RuntimeError(
                    "enable_activation_offloading should only be True when enable_activation_checkpointing is True"
                )
        elif (
            self._enable_activation_checkpointing
            and cfg.checkpointer.model_type != "LLAMA3_VISION"
        ):
            utils.log_rank_zero(
                log,
                "Hint: enable_activation_checkpointing is True, but enable_activation_offloading isn't. "
                "Enabling activation offloading should reduce memory further.",
            )

        # These are public properties which are updated by the checkpoint loader
        # when ``resume_from_checkpoint`` is `True` or validated in tests
        self.seed = training.set_seed(
            seed=cfg.seed, debug_mode=cfg.get("cudnn_deterministic_mode", None)
        )
        self.epochs_run = 0
        self.total_epochs = cfg.epochs
        self.max_steps_per_epoch = cfg.max_steps_per_epoch
        self.global_step = 0

    def _update_recipe_state(self, ckpt_dict: Dict[str, Any]) -> None:
        """
        Updates the recipe state from checkpoint.
        """
        try:
            self.epochs_run = ckpt_dict[training.EPOCHS_KEY]

            # on mismatch, warn the user and prevent the override
            if self.seed != ckpt_dict[training.SEED_KEY]:
                warn(
                    message=(
                        "Config value for seed does not match the checkpoint value, "
                        f"using the checkpoint value: {ckpt_dict[training.SEED_KEY]}"
                    )
                )
                self.seed = ckpt_dict[training.SEED_KEY]
            if self.max_steps_per_epoch != ckpt_dict[training.MAX_STEPS_KEY]:
                warn(
                    message=(
                        "Config value for max_steps_per_epoch does not match the checkpoint value, "
                        f"using the checkpoint value: {ckpt_dict[training.MAX_STEPS_KEY]}"
                    )
                )
                self.max_steps_per_epoch = ckpt_dict[training.MAX_STEPS_KEY]

            # on mismatch, warn the user but allow the override
            if self.total_epochs != ckpt_dict[training.TOTAL_EPOCHS_KEY]:
                warn(
                    message=(
                        "Config value for total_epochs does not match the checkpoint value, "
                        f"using the config value: {self.total_epochs}"
                    )
                )

        except KeyError as e:
            raise KeyError(
                "Checkpoint does not contain the required keys needed for updating recipe state. "
                "Are you sure you passed in the right recipe checkpoint?"
            ) from e

    def setup(self, cfg: DictConfig) -> None:
        """
        Setup the recipe. This includes training state (if resume_from_checkpoint is True),
        model, tokenizer, loss, optimizer, lr scheduler, sampler, and dataloader.
        """
        if self.fsdp_cpu_offload:
            # Utilize all available CPU cores for intra-op parallelism. This provides ~2x
            # speed up when benchmarking fused AdamW on CPU
            training.set_torch_num_threads()

        if self._is_rank_zero:
            self._metric_logger = config.instantiate(cfg.metric_logger)
            # log config with parameter override
            self._metric_logger.log_config(cfg)

        # Load the base model
        checkpoint_dict = self._checkpoint_client.load_base_checkpoint()

        self._compile = cfg.get("compile", False)
        self._model = self._setup_model(
            cfg_model=cfg.model,
            enable_activation_checkpointing=self._enable_activation_checkpointing,
            enable_activation_offloading=self._enable_activation_offloading,
            custom_sharded_layers=cfg.get("custom_sharded_layers", None),
            fsdp_cpu_offload=self.fsdp_cpu_offload,
            reshard_after_forward=cfg.get("fsdp_reshard_after_forward", True),
            model_state_dict=checkpoint_dict[training.MODEL_KEY],
            ac_mode=cfg.get("ac_mode", None),
            ac_option=cfg.get("ac_option", None),
        )
        self._tokenizer = config.instantiate(cfg.tokenizer)
        
        # Extend tokenizer with special tokens if configured
        utils.log_rank_zero(log, f"🔍 Checking tool_call_special_tokens configuration...")
        utils.log_rank_zero(log, f"🔍 hasattr(cfg, 'tool_call_special_tokens'): {hasattr(cfg, 'tool_call_special_tokens')}")
        if hasattr(cfg, 'tool_call_special_tokens'):
            utils.log_rank_zero(log, f"🔍 cfg.tool_call_special_tokens value: {cfg.tool_call_special_tokens}")
            utils.log_rank_zero(log, f"🔍 cfg.tool_call_special_tokens is truthy: {bool(cfg.tool_call_special_tokens)}")
        
        if hasattr(cfg, 'tool_call_special_tokens') and cfg.tool_call_special_tokens:
            utils.log_rank_zero(log, f"✅ Tool call special tokens configuration detected!")
            vocab_size_before = len(self._tokenizer.encoder)
            utils.log_rank_zero(log, f"\n📊 Vocabulary size before extension: {vocab_size_before}")
            
            added_tokens = utils.extend_tokenizer_if_needed(
                self._tokenizer, cfg.tool_call_special_tokens, self._model, self._device, self._dtype
            )
            utils.log_rank_zero(log, f"🔧 extend_tokenizer_if_needed returned: {added_tokens}")
            
            vocab_size_after = len(self._tokenizer.encoder)
            utils.log_rank_zero(log, f"\n📊 Vocabulary size after extension: {vocab_size_after}")

            # Log sample tokenization to show how data is tokenized with special tokens
            sample_data = {
                "messages": [
                    {
                        "role": "user",
                        "content": "This is user message"
                    },
                    {
                        "content": "Let's add the import for `inspect`:\n\n<function=str_replace_editor>\n<parameter=command>str_replace</parameter>\n<parameter=path>/testbed/astropy/coordinates/sky_coordinate.py</parameter>\n<parameter=old_str>\nimport copy\nimport operator\nimport re\nimport warnings\n</parameter>\n<parameter=new_str>\nimport copy\nimport inspect\nimport operator\nimport re\nimport warnings\n</parameter>\n</function>",
                        "role": "assistant"
                    },
                    {
                        "role": "user",
                        "content": "<tool_response>\nThe file `pandas/io/stata.py` has been updated successfully.\n</tool_response>"
                    },
                    {
                        "role": "assistant",
                        "content": "Let's look at the implementation in conan/tools/files/files.py:\n\n<function=str_replace_editor>\n<parameter=command>view</parameter>\n<parameter=path>/testbed/conan/tools/files/files.py</parameter>\n</function>"
                    }
                ]
            }

            utils.log_rank_zero(log, "\n" + "="*80)
            utils.log_rank_zero(log, "SAMPLE TOKENIZATION WITH SPECIAL TOKENS")
            utils.log_rank_zero(log, "="*80)
            utils.log_rank_zero(log, "\nOriginal sample data:")
            for i, msg in enumerate(sample_data["messages"]):
                utils.log_rank_zero(log, f"Message {i} ({msg['role']}): {repr(msg['content'])}")
            
            try:
                # Create a sample transform to demonstrate tokenization
                from torchtune.data._messages import OpenAIToMessages
                from torchtune.datasets._agent_sft import AgentSFTTransform
                
                message_transform = OpenAIToMessages(
                    train_on_input=False,
                    column_map={"messages": "messages"},
                )
                
                agent_transform = AgentSFTTransform(
                    message_transform=message_transform,
                    model_transform=self._tokenizer,
                    enable_token_classification=True,
                )
                
                # Apply transform to sample data
                result = agent_transform(sample_data)
                
                # Extract information
                tokens = result["tokens"]
                labels = result.get("labels", [])
                reasoning_mask = result.get("reasoning_mask", [])
                tool_call_mask = result.get("tool_call_mask", [])
                
                utils.log_rank_zero(log, f"\nTotal tokens: {len(tokens)}")
                utils.log_rank_zero(log, f"Labels length: {len(labels)}")
                utils.log_rank_zero(log, f"Reasoning mask length: {len(reasoning_mask)}")
                utils.log_rank_zero(log, f"Tool call mask length: {len(tool_call_mask)}")
                
                # Show detailed token analysis (first 50 tokens to avoid too much output)
                utils.log_rank_zero(log, f"\n{'Index':<6} {'Token ID':<10} {'Token Text':<25} {'Label':<10} {'Reasoning':<10} {'Tool Call':<10}")
                utils.log_rank_zero(log, "-" * 80)
                
                for i in range(len(tokens)):
                    token_id = tokens[i]
                    
                    # Decode individual token
                    token_text = self._tokenizer.decode([token_id], skip_special_tokens=False)
                    token_text = repr(token_text)
                    
                    label = labels[i] if i < len(labels) else "N/A"
                    reasoning = reasoning_mask[i] if i < len(reasoning_mask) else "N/A"
                    tool_call = tool_call_mask[i] if i < len(tool_call_mask) else "N/A"
                    
                    utils.log_rank_zero(log, f"{i:<6} {token_id:<10} {token_text:<25} {label:<10} {reasoning:<10} {tool_call:<10}")
                
                # Summary statistics
                utils.log_rank_zero(log, "\n" + "="*80)
                utils.log_rank_zero(log, "SUMMARY STATISTICS")
                utils.log_rank_zero(log, "="*80)
                
            except Exception as e:
                utils.log_rank_zero(log, f"Error during sample tokenization logging: {e}")
                import traceback
                utils.log_rank_zero(log, traceback.format_exc())
            
            utils.log_rank_zero(log, "="*80)

        self._optimizer = self._setup_optimizer(
            cfg_optimizer=cfg.optimizer,
            optimizer_in_bwd=self._optimizer_in_bwd,
            opt_state_dict=(
                checkpoint_dict[training.OPT_KEY]
                if training.OPT_KEY in checkpoint_dict
                else None
            ),
        )

        if self._resume_from_checkpoint:
            # If async checkpointing is enabled, intermediate checkpoints are saved asynchronously
            # using the DistributedCheckpointer.
            # Therefore the recipe needs to load the distributed checkpoint to restore the training
            # progress.
            if self._enable_async_checkpointing:
                try:
                    checkpoint_dict = (
                        self._checkpoint_client.load_distributed_checkpoint(
                            self._model,
                            (
                                self._optim_ckpt_wrapper
                                if self._optimizer_in_bwd
                                else self._optimizer
                            ),
                        )
                    )
                except Exception as e:
                    log.warning(
                        f"Failed to load distributed checkpoint: {e}. Training will start from the base checkpoint."
                    )

            # Update the recipe state from the checkpoint state dict.
            self._update_recipe_state(checkpoint_dict)

        # initialize loss
        self._loss_fn = config.instantiate(cfg.loss)

        if self._compile:
            training.compile_loss(self._loss_fn, verbose=self._is_rank_zero)

        if self._loss_fn.__class__.__name__ == "CEWithChunkedOutputLoss":
            # set num_output_chunks for model
            self._model.set_num_output_chunks(self._loss_fn.num_output_chunks)

        utils.log_rank_zero(log, "Loss is initialized.")

        # sampler and dataloader depend on the tokenizer and loss_fn and should be
        # setup after both of these are initialized
        collate_name = cfg.get("collate_fn", "torchtune.data.padded_collate_agent_sft")
        self._dataloader = self._setup_data(
            cfg_dataset=cfg.dataset,
            shuffle=cfg.shuffle,
            batch_size=cfg.batch_size,
            collate_fn=collate_name,
        )
        
        # Setup validation dataloader if validation is enabled
        if self._enable_validation:
            self._val_dataloader = self._setup_data(
                cfg_dataset=cfg.dataset_val,
                shuffle=False,
                batch_size=self._batch_size_val,
                collate_fn=collate_name,
            )
            utils.log_rank_zero(log, "Validation dataloader is set up.")
        else:
            self._val_dataloader = None

        # Log one example after tokenization
        train_example = self._dataloader.dataset[0]
        # input_ids = self._tokenizer(train_example)['tokens']
        decoded_train_example = self._tokenizer.decode(train_example['tokens'], skip_special_tokens=False)
        utils.log_rank_zero(log, decoded_train_example)

        # Finally update the recipe state which can only be correctly set after all of the
        # other components have been initialized and updated.
        #
        # Number of training steps in each epoch depends on the number of batches produced
        # by the dataloader, the max_steps_per_epoch param set by the user and the
        # gradient_accumulation_steps param. This value is used for logging and tracking
        # training state. The computation should happen after the dataloader has been setup
        self._steps_per_epoch = (
            len(self._dataloader) // self._gradient_accumulation_steps
        )
        if (
            self.max_steps_per_epoch is not None
            and self.max_steps_per_epoch < self._steps_per_epoch
        ):
            self._steps_per_epoch = self.max_steps_per_epoch
        self.global_step = self.epochs_run * self._steps_per_epoch

        # Setup lr scheduler
        self._lr_scheduler = self._setup_lr_scheduler(
            cfg_lr_scheduler=cfg.get("lr_scheduler", None),
            num_training_steps=self.total_epochs * self._steps_per_epoch,
            last_epoch=self.global_step - 1,
        )

        # Set up profiler, returns DummyProfiler (nullcontext object with no-op `step` method)
        # if cfg is missing profiler key or if `cfg.profiler.enabled = False`
        self._profiler = self._setup_profiler(cfg.get(PROFILER_KEY, None))

        # Used to ignore labels for loss computation
        self.ignore_labels_cache = torch.full(
            (cfg.batch_size, 1), self._loss_fn.ignore_index, device=self._device
        )

    def _setup_lr_scheduler(
        self,
        cfg_lr_scheduler: Optional[DictConfig],
        num_training_steps: int,
        last_epoch: int,
    ) -> Optional[Optimizer]:
        """
        Set up the learning rate scheduler based on the provided configuration.
        It supports both standard optimization and optimizer-in-backward cases.

        Args:
            cfg_lr_scheduler (Optional[DictConfig]): The learning rate scheduler configuration.
            num_training_steps (int): The total number of training steps.
            last_epoch (int): The index of the last epoch.

        Returns:
            lr_scheduler (Optional[Optimizer]): The learning rate scheduler.
        """
        if cfg_lr_scheduler is None:
            if self._is_rank_zero:
                log.info(
                    "No learning rate scheduler configured. Using constant learning rate."
                )
            return None

        if self._optimizer_in_bwd:
            # Use the first optimizer from the wrapper to represent the learning rate
            optimizer = next(iter(self._optim_ckpt_wrapper.optim_map.values()))
        else:
            # Standard case: use the single optimizer
            optimizer = self._optimizer

        # Instantiate the learning rate scheduler
        lr_scheduler = config.instantiate(
            cfg_lr_scheduler,
            optimizer,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
        )

        if self._optimizer_in_bwd:
            # Modify the scheduler for optimizer_in_bwd case
            self._optim_ckpt_wrapper.set_lr_scheduler(lr_scheduler)

        if self._is_rank_zero:
            log.info("Learning rate scheduler is initialized.")

        return lr_scheduler

    def _setup_profiler(
        self, cfg_profiler: Optional[DictConfig] = None
    ) -> Union[torch.profiler.profile, DummyProfiler]:
        """
        Parses the `profiler` section of top-level `cfg` and sets up profiler
        """
        # Missing profiler section in config, assume disabled
        if cfg_profiler is None:
            cfg_profiler = DictConfig({"enabled": False})

        # Check that component is included and set correctly
        if cfg_profiler.get("_component_", None) is None:
            cfg_profiler["_component_"] = "torchtune.training.setup_torch_profiler"
        else:
            assert (
                cfg_profiler.get("_component_")
                == "torchtune.training.setup_torch_profiler"
            ), "Only torch profiler supported currently: component must be `torchtune.training.setup_torch_profiler`"

        profiler, profiler_cfg = config.instantiate(cfg_profiler)

        utils.log_rank_zero(
            log, f" Profiler config after instantiation: {profiler_cfg}"
        )
        if self._is_rank_zero:
            self.profiler_profile_memory = profiler_cfg.get("profile_memory", False)
            if profiler_cfg["enabled"]:
                self.profiler_wait_steps = profiler_cfg["wait_steps"]
                self.profiler_warmup_steps = profiler_cfg["warmup_steps"]
                self.profiler_active_steps = profiler_cfg["active_steps"]

        return profiler

    def _setup_model(
        self,
        cfg_model: DictConfig,
        enable_activation_checkpointing: bool,
        enable_activation_offloading: bool,
        fsdp_cpu_offload: bool,
        reshard_after_forward: bool,
        model_state_dict: Dict[str, Any],
        custom_sharded_layers: Optional[List[str]] = None,
        ac_mode: Optional[str] = None,
        ac_option: Optional[int] = None,
    ) -> nn.Module:
        """
        Model initialization has some important considerations:
           a. To minimize GPU peak memory, we initialize the model on meta device with
              the right dtype
           b. All ranks calls ``load_state_dict`` without peaking CPU RAMs since
              full state dicts are loaded with ``torch.load(mmap=True)``
        """

        utils.log_rank_zero(
            log,
            "Distributed training is enabled. Instantiating model and loading checkpoint on Rank 0 ...",
        )
        init_start = time.perf_counter()

        with training.set_default_dtype(self._dtype), torch.device("meta"):
            model = config.instantiate(cfg_model)

        if self._compile:
            training.compile_model(model, verbose=self._is_rank_zero)

        device_mesh = init_device_mesh(
            self._device.type,
            mesh_shape=(self.data_parallel_dim, self.tensor_parallel_dim),
            mesh_dim_names=("dp", "tp"),
        )
        self.dp_size = device_mesh["dp"].size()
        self.dp_rank = device_mesh["dp"].get_local_rank()

        # Apply tensor parallelism to the model
        if self.tensor_parallel_dim > 1:
            # Use the local number (num_heads, num_kv_heads, embed_dim) to account for tensor parallel
            model = training.prepare_mha_for_tp(model, device_mesh["tp"])
            parallelize_module(
                model,
                device_mesh["tp"],
                parallelize_plan=self.tensor_parallel_plan,
            )

        # We currently have two versions of activation checkpointing in this recipe
        # for testing and BC purposes. ``enable_activation_checkpointing`` controls
        # the older version of AC and this behavior is unchanged
        # ac_mode and ac_option together control selective AC. This is only enabled
        # when these are set AND ``enable_activation_checkpointing`` is set to False
        # We'll clean this up as soon as testing of AC is complete
        if (not enable_activation_checkpointing) and (ac_mode is not None):
            apply_selective_activation_checkpointing(
                model,
                ac_mode,
                ac_option,
            )

        # original activation checkpointing (full) - flip the condition above
        if enable_activation_checkpointing and ac_mode is None:
            training.set_activation_checkpointing(
                model, auto_wrap_policy={modules.TransformerSelfAttentionLayer}
            )

        # Apply Fully Sharded Data Parallelism to the model
        if self.data_parallel_dim > 1:
            fsdp_shard_conditions = [
                partial(
                    training.get_shard_conditions,
                    names_to_match=custom_sharded_layers,
                )
            ]
            training.shard_model(
                model=model,
                shard_conditions=fsdp_shard_conditions,
                cpu_offload=fsdp_cpu_offload,
                reshard_after_forward=reshard_after_forward,
                dp_mesh=device_mesh["dp"],
            )

        with training.set_default_dtype(self._dtype), self._device:
            for m in model.modules():
                # RoPE is not covered in state dict
                if hasattr(m, "rope_init"):
                    m.rope_init()

        # This method will convert the full model state dict into a sharded state
        # dict and load into the model
        training.load_from_full_model_state_dict(
            model,
            model_state_dict,
            self._device,
            strict=True,
            cpu_offload=fsdp_cpu_offload,
        )

        # activation offloading
        self.activations_handling_ctx = training.get_act_offloading_ctx_manager(
            model, enable_activation_offloading
        )

        # Ensure no params and buffers are on meta device
        training.validate_no_params_on_meta_device(model)

        utils.log_rank_zero(
            log,
            f"Instantiating model and loading checkpoint took {time.perf_counter() - init_start:.2f} secs",
        )

        if self._is_rank_zero:
            memory_stats = training.get_memory_stats(device=self._device)
            training.log_memory_stats(memory_stats)

        # synchronize before training begins
        torch.distributed.barrier()

        return model

    def _setup_optimizer(
        self,
        cfg_optimizer: DictConfig,
        optimizer_in_bwd: bool = False,
        opt_state_dict: Optional[Dict[str, Any]] = None,
    ) -> Optional[Optimizer]:
        if optimizer_in_bwd:
            # Maintain a dict of optims for every parameter.
            optim_dict = {
                param: config.instantiate(cfg_optimizer, [param])
                for param in self._model.parameters()
            }

            # Register optimizer step hooks on the model to run optimizer in backward.
            training.register_optim_in_bwd_hooks(
                model=self._model, optim_dict=optim_dict
            )
            # Create a wrapper for checkpoint save/load of optimizer states when running in backward.
            self._optim_ckpt_wrapper = training.create_optim_in_bwd_wrapper(
                model=self._model, optim_dict=optim_dict
            )
            # Load optimizer states for each param. If optimizer states are being restored in an optimizer in
            # backward run, these need to have been saved with the same setting. Cannot restore from runs that
            # did not use optimizer in backward.
            if opt_state_dict is not None:
                for param in opt_state_dict.keys():
                    try:
                        training.load_from_full_optimizer_state_dict(
                            self._model,
                            self._optim_ckpt_wrapper.optim_map[param],
                            opt_state_dict[param],
                            self._device,
                        )
                    except BaseException as e:
                        raise RuntimeError(
                            "Failed loading in-backward optimizer checkpoints."
                            "Please make sure run being restored from was using in-backward optimizer."
                        ) from e
            utils.log_rank_zero(log, "In-backward optimizers are set up.")
            return None
        else:
            optimizer = config.instantiate(cfg_optimizer, self._model.parameters())
            if opt_state_dict:
                training.load_from_full_optimizer_state_dict(
                    self._model,
                    optimizer,
                    opt_state_dict,
                    self._device,
                )

            utils.log_rank_zero(log, "Optimizer is initialized.")
            return optimizer

    def _setup_data(
        self,
        cfg_dataset: DictConfig,
        shuffle: bool,
        batch_size: int,
        collate_fn: str,
        dataloader_state_dict: Optional[Dict[str, Any]] = None,
    ) -> StatefulDataLoader:
        """
        All data related setup happens here. This recipe supports enhanced agent SFT datasets
        that include token classification for detailed metrics tracking.
        """
        if isinstance(cfg_dataset, ListConfig):
            datasets = []
            for single_cfg_dataset in cfg_dataset:
                # Use AgentSFTDataset if it's configured for agent training
                if single_cfg_dataset.get("_component_", "").endswith("AgentSFTDataset"):
                    ds = config.instantiate(
                        single_cfg_dataset,
                        self._tokenizer,
                    )
                else:
                    # Fallback to regular SFT dataset for compatibility
                    ds = config.instantiate(single_cfg_dataset, self._tokenizer)
                datasets.append(ds)
            ds = ConcatDataset(datasets=datasets)
            packed = getattr(ds, "packed", False)
        else:
            # Use AgentSFTDataset if configured for agent training
            if cfg_dataset.get("_component_", "").endswith("AgentSFTDataset"):
                ds = config.instantiate(
                    cfg_dataset,
                    self._tokenizer,
                )
            else:
                # Fallback to regular SFT dataset for compatibility
                ds = config.instantiate(cfg_dataset, self._tokenizer)
            packed = cfg_dataset.get("packed", False)

        # Instantiate collate_fn
        if "left_pad_sequence" in collate_fn:
            raise RuntimeError("left_pad_sequence collator is only for inference.")
        collate_fn = _get_component_from_path(collate_fn)

        sampler = StatefulDistributedSampler(
            ds, num_replicas=self.dp_size, rank=self.dp_rank, shuffle=shuffle
        )
        dataloader = StatefulDataLoader(
            dataset=ds,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=(
                partial(
                    collate_fn,
                    padding_idx=self._tokenizer.pad_id,
                    ignore_idx=self._loss_fn.ignore_index,
                )
                if not packed
                else padded_collate_packed
            ),
            # dropping last avoids shape issues with compile + flex attention
            drop_last=True,
        )

        return dataloader

    def _compute_detailed_metrics(
        self, 
        logits: torch.Tensor, 
        labels: torch.Tensor, 
        loss: torch.Tensor,
        batch: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Compute detailed metrics if enabled.
        """
        if not self._enable_detailed_metrics:
            return {}
            
        # Extract token classification masks if available
        reasoning_mask = batch.get("reasoning_mask", None)
        tool_call_mask = batch.get("tool_call_mask", None)
        
        metrics = self._metrics_tracker.compute_metrics(
            logits=logits,
            labels=labels,
            loss=loss,
            reasoning_mask=reasoning_mask,
            tool_call_mask=tool_call_mask,
        )
        
        return metrics

    def validate(self) -> float:
        """
        Run validation loop and return average validation loss.
        """
        if not self._enable_validation:
            return 0.0
            
        self._model.eval()
        
        total_val_loss = torch.tensor(0.0, device=self._device, dtype=self._dtype)
        total_val_tokens = torch.tensor(0, device=self._device, dtype=torch.long)
        
        # Accumulate detailed metrics for validation
        val_detailed_metrics = {}
        val_metrics_count = 0
        
        # Create validation progress bar - only show on rank 0
        val_pbar = tqdm(
            total=len(self._val_dataloader),
            desc="Validation",
            disable=not self._is_rank_zero,
            leave=False,
            ncols=100,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self._val_dataloader):
                utils.batch_to_device(batch, self._device)
                
                # Calculate the number of unmasked tokens in the current batch
                current_num_tokens = (
                    batch["labels"] != self._loss_fn.ignore_index
                ).sum()
                
                # Store batch size before popping labels
                batch_size = batch["labels"].shape[0]
                
                labels = batch.pop("labels")
                
                # Pop masks that are used for metrics but not for model forward
                reasoning_mask = batch.pop("reasoning_mask", None)
                tool_call_mask = batch.pop("tool_call_mask", None)
                
                with self.activations_handling_ctx:
                    logits = self._model(**batch)

                # Store original labels for metrics computation (before shifting)
                original_labels = labels.clone()
                
                # Reconstruct full logits tensor if chunked for metrics computation
                logits_for_metrics = None
                if isinstance(logits, list):
                    # Concatenate chunked logits back to original shape
                    logits_for_metrics = torch.cat(logits, dim=1)
                    # print(f"Concatenated logits shape: {logits_for_metrics.shape}")
                    # print(f"Original labels shape: {original_labels.shape}")
                else:
                    logits_for_metrics = logits

                # Shift labels to compute loss
                labels = torch.hstack(
                    (labels[..., 1:], self.ignore_labels_cache[: labels.shape[0]])
                )

                if not isinstance(logits, list):
                    labels = labels.reshape(-1)
                    logits = logits.reshape(-1, logits.size(-1))

                # Compute validation loss (normalized by number of tokens)
                val_loss = self._loss_fn(logits, labels) * current_num_tokens

                total_val_loss += val_loss
                total_val_tokens += current_num_tokens

                # Compute detailed metrics for validation
                if self._enable_detailed_metrics and batch_idx % self._log_detailed_metrics_every_n_steps == 0 and logits_for_metrics is not None:
                    # Add masks back to batch for metrics computation
                    if reasoning_mask is not None:
                        batch["reasoning_mask"] = reasoning_mask
                    if tool_call_mask is not None:
                        batch["tool_call_mask"] = tool_call_mask
                    
                    batch_metrics = self._compute_detailed_metrics(
                        logits_for_metrics, original_labels, val_loss / current_num_tokens, batch
                    )
                    
                    # Accumulate metrics
                    for key, value in batch_metrics.items():
                        if key in val_detailed_metrics:
                            val_detailed_metrics[key] += value
                        else:
                            val_detailed_metrics[key] = value
                    val_metrics_count += 1
                
                # Update progress bar with current validation loss (only on rank 0)
                if self._is_rank_zero:
                    current_val_loss = val_loss.item() / current_num_tokens.item() if current_num_tokens.item() > 0 else 0.0
                    val_pbar.set_postfix({"val_loss": f"{current_val_loss:.4f}"})
                    val_pbar.update(1)
                
                # free logits to save memory
                del logits
        
        # Close the progress bar (only on rank 0)
        if self._is_rank_zero:
            val_pbar.close()
        
        # Aggregate across all ranks
        torch.distributed.all_reduce(total_val_loss)
        torch.distributed.all_reduce(total_val_tokens)
        
        # Calculate average validation loss
        avg_val_loss = total_val_loss.item() / total_val_tokens.item() if total_val_tokens.item() > 0 else 0.0
        
        # Average detailed metrics and log them
        if self._enable_detailed_metrics and val_metrics_count > 0 and self._is_rank_zero:
            avg_val_detailed_metrics = {
                f"val_{key}": value / val_metrics_count 
                for key, value in val_detailed_metrics.items()
            }
            self._metric_logger.log_dict(avg_val_detailed_metrics, step=self.global_step)
        
        # Set model back to training mode
        self._model.train()
        
        # Update the last validation loss for display in training progress
        self._last_val_loss = avg_val_loss
        
        return avg_val_loss

    def train(self) -> None:
        """
        The core training loop with detailed metrics tracking.
        """
        # clean up before training begins
        training.cleanup_before_training()

        # zero out the gradients before starting training
        if not self._optimizer_in_bwd:
            self._optimizer.zero_grad()
        else:
            for opt in self._optim_ckpt_wrapper.optim_map.values():
                opt.zero_grad()

        # Initialize tokens count and running loss (for grad accumulation)
        t0 = time.perf_counter()
        running_loss = 0
        num_tokens = 0

        self._profiler.start()
        # self.epochs_run should be non-zero when we're resuming from a checkpoint
        for curr_epoch in range(self.epochs_run, self.total_epochs):
            pbar = tqdm(total=self._steps_per_epoch, disable=not self._is_rank_zero)
            self._dataloader.sampler.set_epoch(curr_epoch)
            for idx, batch in enumerate(self._dataloader):
                # Start tracking CUDA memory for active steps for just the first epoch
                if (
                    self._is_rank_zero
                    and curr_epoch == 0
                    and self.profiler_profile_memory
                    and idx == self.profiler_wait_steps + self.profiler_warmup_steps
                    and self._device.type == "cuda"
                ):
                    torch.cuda.memory._record_memory_history()
                utils.batch_to_device(batch, self._device)

                # Calculate the number of unmasked tokens in the current batch
                # and increment the total number of tokens seen in the step
                current_num_tokens = (
                    batch["labels"] != self._loss_fn.ignore_index
                ).sum()
                num_tokens += current_num_tokens

                # Store batch size before popping labels
                batch_size = batch["labels"].shape[0]

                # Shape [b, s], needed for the loss not the model
                labels = batch.pop("labels")
                
                # Pop masks that are used for metrics but not for model forward
                reasoning_mask = batch.pop("reasoning_mask", None)
                tool_call_mask = batch.pop("tool_call_mask", None)

                with self.activations_handling_ctx:
                    logits = self._model(**batch)
                
                # Store original labels for metrics computation (before shifting)
                original_labels = labels.clone()
                
                # Reconstruct full logits tensor if chunked for metrics computation
                logits_for_metrics = None
                if isinstance(logits, list):
                    # Concatenate chunked logits back to original shape
                    logits_for_metrics = torch.cat(logits, dim=1)
                    # print(f"Concatenated logits shape: {logits_for_metrics.shape}")
                    # print(f"Original labels shape: {original_labels.shape}")
                else:
                    logits_for_metrics = logits
    
                # Shift labels to compute loss
                # equivalent to doing labels[..., 1:] and logits[..., :-1, :]
                # But this way we dont need to slice the logits. We just add an ignore index to labels.
                labels = torch.hstack(
                    (labels[..., 1:], self.ignore_labels_cache[: labels.shape[0]])
                )
                
                if not isinstance(logits, list):
                    labels = labels.reshape(-1)
                    logits = logits.reshape(-1, logits.size(-1))

                # Compute loss
                # Loss is normalized by default so we multiply by the number of tokens
                # This way we can normalize by the total number of tokens if we're accumulating gradients
                current_loss = self._loss_fn(logits, labels) * current_num_tokens

                # Compute detailed metrics if enabled
                if (
                    self._enable_detailed_metrics 
                    and self.global_step % self._log_detailed_metrics_every_n_steps == 0
                    and logits_for_metrics is not None
                ):
                    # Add masks back to batch for metrics computation
                    if reasoning_mask is not None:
                        batch["reasoning_mask"] = reasoning_mask
                    if tool_call_mask is not None:
                        batch["tool_call_mask"] = tool_call_mask
                    
                    detailed_metrics = self._compute_detailed_metrics(
                        logits_for_metrics, original_labels, current_loss / current_num_tokens, batch
                    )
                    
                    if detailed_metrics and self._is_rank_zero:
                        # Add train prefix to metrics
                        train_detailed_metrics = {
                            f"train_{key}": value for key, value in detailed_metrics.items()
                        }
                        self._metric_logger.log_dict(train_detailed_metrics, step=self.global_step)

                # free logits otherwise it peaks backward memory
                del logits

                running_loss += current_loss

                # For optimizer in backward, we need to normalize before calling backward
                # This case and gradient accumulation are mutually exclusive
                if self._optimizer_in_bwd:
                    torch.distributed.all_reduce(num_tokens)
                    torch.distributed.all_reduce(running_loss)

                    # We multiply by world_size to undo FSDP2 gradient normalization.
                    current_loss = current_loss * (self.world_size / num_tokens)

                current_loss.backward()

                # Step with optimizer
                if (idx + 1) % self._gradient_accumulation_steps == 0:
                    if not self._optimizer_in_bwd:
                        # Get total number of tokens across all ranks to normalize gradients
                        torch.distributed.all_reduce(num_tokens)
                        # This will ensure that the logged loss matches what we're optimizing
                        torch.distributed.all_reduce(running_loss)
                        # Manually scale the gradients from unnormalized loss by total # of tokens
                        # We multiply by world_size to undo FSDP2 gradient normalization.
                        training.scale_grads(self._model, self.world_size / num_tokens)
                        if self._clip_grad_norm is not None:
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                self._model.parameters(),
                                max_norm=float(self._clip_grad_norm),
                            )
                            # If sharded, collect the DTensor here
                            if isinstance(grad_norm, DTensor):
                                grad_norm = grad_norm.full_tensor()
                        self._optimizer.step()
                        self._optimizer.zero_grad(set_to_none=True)

                    # Update the number of steps when the weights are updated
                    self.global_step += 1

                    # Step the learning rate scheduler
                    if self._lr_scheduler is not None:
                        self._lr_scheduler.step()

                    loss_to_log = running_loss.item() / num_tokens
                    pbar.update(1)
                    
                    # Create progress description with validation loss if available
                    desc = f"{curr_epoch + 1}|{self.global_step}|Loss: {loss_to_log:.4f}"
                    if self._last_val_loss is not None:
                        desc += f"|Val: {self._last_val_loss:.4f}"
                    pbar.set_description(desc)

                    # Log per-step metrics
                    if (
                        self.global_step % self._log_every_n_steps == 0
                        and self._is_rank_zero
                    ):
                        time_per_step = time.perf_counter() - t0
                        log_dict = {
                            "loss": loss_to_log,
                            "lr": get_lr(
                                (
                                    self._optimizer
                                    if not self._optimizer_in_bwd
                                    else self._optim_ckpt_wrapper
                                ),
                            ),
                            "tokens_per_second_per_gpu": num_tokens
                            / (time_per_step * self.world_size),
                        }
                        if self._log_peak_memory_stats:
                            log_dict.update(
                                training.get_memory_stats(device=self._device)
                            )
                        if self._clip_grad_norm is not None:
                            log_dict.update({"grad_norm": grad_norm})
                        self._metric_logger.log_dict(
                            log_dict,
                            step=self.global_step,
                        )
                    
                    # Run validation if enabled and at the right interval
                    # Note: All ranks must participate in validation due to all_reduce operations
                    if (
                        self._enable_validation 
                        and self.global_step % self._run_val_every_n_steps == 0
                    ):
                        utils.log_rank_zero(
                            log,
                            f"Running validation at step {self.global_step}..."
                        )
                        val_loss = self.validate()
                        if self._is_rank_zero:
                            val_log_dict = {"val_loss": val_loss}
                            self._metric_logger.log_dict(
                                val_log_dict,
                                step=self.global_step,
                            )
                            utils.log_rank_zero(
                                log,
                                f"Step {self.global_step}: Validation Loss: {val_loss:.4f}"
                            )

                    # Reset running stats for the next step
                    running_loss = 0
                    num_tokens = 0
                    t0 = time.perf_counter()

                    # Stop tracking CUDA memory now that active steps are complete
                    if (
                        self._is_rank_zero
                        and curr_epoch == 0
                        and self.profiler_profile_memory
                        and idx
                        == self.profiler_wait_steps
                        + self.profiler_warmup_steps
                        + self.profiler_active_steps
                        and self._device.type == "cuda"
                    ):
                        torch.cuda.memory._record_memory_history(enabled=None)

                    # Step profiler
                    # Note that this is called within gradient accumulation block, hence
                    # will include multiple forward / backward passes if gradient accumulation > 1
                    self._profiler.step()

                if (
                    (idx + 1) // self._gradient_accumulation_steps
                ) == self.max_steps_per_epoch:
                    break

            self.epochs_run += 1
            
            # Run validation at the end of each epoch if enabled
            # Note: All ranks must participate in validation due to all_reduce operations
            if self._enable_validation:
                utils.log_rank_zero(
                    log,
                    f"Running end-of-epoch validation for epoch {curr_epoch + 1}..."
                )
                val_loss = self.validate()
                if self._is_rank_zero:
                    val_log_dict = {"val_loss": val_loss}
                    self._metric_logger.log_dict(
                        val_log_dict,
                        step=self.global_step,
                    )
                    utils.log_rank_zero(
                        log,
                        f"Epoch {curr_epoch + 1} completed: Validation Loss: {val_loss:.4f}"
                    )
            
            self._checkpoint_client.save_checkpoint(
                model=self._model,
                optimizer=(
                    self._optimizer
                    if not self._optimizer_in_bwd
                    else self._optim_ckpt_wrapper
                ),
                training_progress=TrainingProgress(
                    seed=self.seed,
                    epochs_run=self.epochs_run,
                    total_epochs=self.total_epochs,
                    max_steps_per_epoch=self.max_steps_per_epoch,
                    dataloader_state_dict=self._dataloader.state_dict(),
                ),
                epoch=curr_epoch,
            )
            
            # Update tokenizer.json with extended tokens after checkpoint save
            if hasattr(self._cfg, 'tool_call_special_tokens') and self._cfg.tool_call_special_tokens:
                tokenizer_json_path = f"{self._output_dir}/epoch_{curr_epoch}/tokenizer.json"
                utils.update_tokenizer_json_added_tokens(
                    self._tokenizer,
                    tokenizer_json_path
                )
                utils.log_rank_zero(log, f"Updated tokenizer.json with extended tokens at {tokenizer_json_path}")

        self._profiler.stop()

    def cleanup(self) -> None:
        if self._is_rank_zero:
            self._metric_logger.close()
        destroy_process_group()


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    """
    Entry point for the agent SFT recipe.

    Configurable parameters are read in the following order:
        - Parameters specified in config (see available configs through ``tune ls``)
        - Overwritten by arguments from the command-line
    """
    config.log_config(recipe_name="AgentSFTRecipeDistributed", cfg=cfg)
    recipe = AgentSFTRecipeDistributed(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())