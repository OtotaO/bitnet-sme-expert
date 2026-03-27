"""Fine-tuning service using Unsloth for efficient LLM fine-tuning."""
import asyncio
import logging
import os
import json
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Callable, Awaitable

from unsloth import FastLanguageModel
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from transformers import (
    TrainingArguments,
    set_seed,
    get_linear_schedule_with_warmup,
    BitsAndBytesConfig
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

# Type aliases
TrainingCallback = Callable[[Dict[str, Any]], Awaitable[None]]


logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for fine-tuning with distributed training support.
    
    Attributes:
        model_name: Name or path of the base model to fine-tune
        dataset_path: Path to training dataset (JSONL format)
        output_dir: Directory to save trained models and checkpoints
        max_seq_length: Maximum sequence length for the model
        batch_size: Batch size per device
        gradient_accumulation_steps: Number of steps to accumulate gradients
        warmup_steps: Number of warmup steps for learning rate scheduler
        num_train_epochs: Number of training epochs
        learning_rate: Initial learning rate
        weight_decay: Weight decay for optimization
        lr_scheduler_type: Learning rate scheduler type (e.g., 'linear', 'cosine')
        logging_steps: Log training metrics every N steps
        save_steps: Save checkpoint every N steps
        eval_steps: Evaluate model every N steps (None for evaluation at epoch end)
        fp16: Use FP16 mixed precision training
        bf16: Use BF16 mixed precision training
        lora_rank: Rank for LoRA adapters
        lora_alpha: Alpha parameter for LoRA
        lora_dropout: Dropout probability for LoRA layers
        max_grad_norm: Maximum gradient norm for gradient clipping
        seed: Random seed for reproducibility
        ddp_find_unused_parameters: Whether to find unused parameters in DDP
        local_rank: Local rank for distributed training (-1 for single GPU)
        world_size: Total number of processes for distributed training
        gradient_checkpointing: Whether to use gradient checkpointing
        fsdp: Whether to use Fully Sharded Data Parallel
        fsdp_config: Configuration for FSDP
    """
    # Model and data
    model_name: str = "unsloth/llama-3-8b-bnb-4bit"
    dataset_path: Optional[str] = None
    output_dir: str = "models/finetuned"
    max_seq_length: int = 2048
    
    # Training hyperparameters
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    num_train_epochs: int = 1
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    
    # Logging and checkpointing
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: Optional[int] = None
    
    # Precision
    fp16: bool = torch.cuda.is_available()
    bf16: bool = not torch.cuda.is_available()
    
    # LoRA config
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    # Optimization
    max_grad_norm: float = 0.3
    seed: int = 42
    
    # Distributed training
    ddp_find_unused_parameters: bool = False
    local_rank: int = -1
    world_size: int = 1
    
    # Memory optimization
    gradient_checkpointing: bool = True
    
    # FSDP (Fully Sharded Data Parallel)
    fsdp: bool = False
    fsdp_config: Optional[Dict[str, Any]] = field(
        default_factory=lambda: {
            "sharding_strategy": "FULL_SHARD",
            "min_num_params": 1e8,
            "cpu_offload": False,
        }
    )
    
    def __post_init__(self):
        """Post-initialization validation."""
        if self.fp16 and self.bf16:
            raise ValueError("Cannot use both FP16 and BF16 precision")
            
        if self.local_rank == -1 and self.world_size > 1:
            logger.warning("world_size > 1 but local_rank not set, using single process training")

class FineTuningService:
    """Service for fine-tuning LLMs using Unsloth with distributed training support.
    
    This service provides both synchronous and asynchronous interfaces for model training,
    with support for distributed training across multiple GPUs/nodes using DDP or FSDP.
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        """Initialize the fine-tuning service.
        
        Args:
            config: Configuration for fine-tuning. If None, uses default values.
        """
        self.config = config or TrainingConfig()
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self._is_initialized = False
        self._training_task = None
        self._stop_training = False
        self._callbacks: List[TrainingCallback] = []
        
        # Set up distributed training if needed
        self._setup_distributed()
        
        # Set random seed for reproducibility
        set_seed(self.config.seed + (self.config.local_rank if self.config.local_rank != -1 else 0))
    
    def _setup_distributed(self):
        """Initialize distributed training if needed."""
        if self.config.local_rank != -1:
            torch.cuda.set_device(self.config.local_rank)
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=self.config.world_size,
                rank=self.config.local_rank
            )
            logger.info(f"Initialized distributed training on rank {self.config.local_rank}")
    
    def add_callback(self, callback: TrainingCallback):
        """Add a callback to be called during training.
        
        Args:
            callback: An async function that takes a dictionary of training metrics
        """
        self._callbacks.append(callback)
    
    async def _notify_callbacks(self, metrics: Dict[str, Any]):
        """Notify all registered callbacks with training metrics."""
        if not self._callbacks:
            return
            
        # Add distributed training info if available
        if self.config.local_rank != -1:
            metrics.update({
                'local_rank': self.config.local_rank,
                'world_size': self.config.world_size
            })
        
        # Run callbacks in parallel
        await asyncio.gather(
            *(callback(metrics) for callback in self._callbacks),
            return_exceptions=True
        )
    
    def stop_training(self):
        """Signal the training to stop at the next opportunity."""
        self._stop_training = True
        if self.trainer:
            self.trainer.should_training_stop = True
    
    def is_training(self) -> bool:
        """Check if training is currently in progress."""
        return self._training_task is not None and not self._training_task.done()
    
    async def wait_for_training(self):
        """Wait for the current training task to complete."""
        if self._training_task and not self._training_task.done():
            await self._training_task
    
    async def cleanup(self):
        """Clean up resources and stop any ongoing training."""
        self.stop_training()
        await self.wait_for_training()
        
        if self.config.local_rank != -1:
            dist.destroy_process_group()
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self._is_initialized = False
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
    
    def load_model(self):
        """Load the base model with Unsloth optimizations."""
        logger.info(f"Loading model: {self.config.model_name}")
        
        # Load model with Unsloth optimizations
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model_name,
            max_seq_length=self.config.max_seq_length,
            dtype=None,  # Auto-detect
            load_in_4bit=True,  # Use 4-bit quantization
            # token = "hf_...",  # Uncomment to use HF token
        )
        
        # Prepare for LoRA fine-tuning
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.config.lora_rank,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            use_gradient_checkpointing=True,
            random_state=self.config.seed,
            max_seq_length=self.config.max_seq_length,
        )
        
        logger.info("Model loaded and prepared for fine-tuning")
    
    def load_dataset(self, dataset_path: Optional[str] = None) -> Dataset:
        """Load and prepare the dataset."""
        dataset_path = dataset_path or self.config.dataset_path
        if not dataset_path:
            raise ValueError("No dataset path provided")
            
        logger.info(f"Loading dataset from {dataset_path}")
        
        # Load dataset (supports JSON, CSV, or DatasetDict)
        if dataset_path.endswith('.json'):
            dataset = Dataset.from_json(dataset_path)
        elif dataset_path.endswith('.csv'):
            dataset = Dataset.from_csv(dataset_path)
        else:
            # Assume it's a dataset name from the Hugging Face Hub
            from datasets import load_dataset
            dataset = load_dataset(dataset_path)
            
            # If it's a DatasetDict, use the 'train' split
            if isinstance(dataset, dict):
                dataset = dataset["train"]
        
        # Example formatting function - adjust based on your dataset
        def format_instruction(example):
            return {"text": f"### Instruction: {example['instruction']}\n### Response: {example['response']}"}
        
        # Apply formatting
        dataset = dataset.map(
            format_instruction,
            remove_columns=[col for col in dataset.column_names if col not in ["text"]],
        )
        
        return dataset
    
    def train(self, dataset: Optional[Dataset] = None):
        """Train the model on the provided dataset."""
        if self.model is None:
            self.load_model()
        
        if dataset is None:
            dataset = self.load_dataset()
        
        logger.info("Starting training...")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            num_train_epochs=self.config.num_train_epochs,
            learning_rate=self.config.learning_rate,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            max_grad_norm=self.config.max_grad_norm,
            save_total_limit=3,
            load_best_model_at_end=True if self.config.eval_steps else False,
            report_to="tensorboard",
            seed=self.config.seed,
        )
        
        # Initialize trainer
        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=self.config.max_seq_length,
            tokenizer=self.tokenizer,
            args=training_args,
            packing=True,  # Pack multiple short examples in the same input sequence
        )
        
        # Start training
        self.trainer.train()
        
        # Save the final model
        self.save_model()
        
        logger.info("Training completed")
    
    def save_model(self, output_dir: Optional[str] = None):
        """Save the fine-tuned model and tokenizer."""
        if self.trainer is None:
            raise ValueError("No training has been performed")
            
        output_dir = output_dir or self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the model
        self.trainer.save_model(output_dir)
        
        # Save tokenizer
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        
        # Save training config
        config_path = os.path.join(output_dir, "training_config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        logger.info(f"Model saved to {output_dir}")
    
    @classmethod
    def from_pretrained(cls, model_path: str):
        """Load a fine-tuned model."""
        # Load config
        config_path = os.path.join(model_path, "training_config.json")
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # Create instance
        config = TrainingConfig(**config_data)
        config.output_dir = model_path  # Override output dir
        instance = cls(config)
        
        # Load model
        instance.model, instance.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=config.max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
        
        return instance
