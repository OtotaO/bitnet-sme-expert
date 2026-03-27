"""Enhanced fine-tuning service using Unsloth for efficient LLM fine-tuning with chat logs."""
import os
import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable, Awaitable, Tuple
from dataclasses import dataclass, field
import torch
import torch.distributed as dist
from datasets import Dataset, load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    TrainingArguments,
    set_seed,
    get_linear_schedule_with_warmup,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments
)
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer

from .chat_log_processor import ChatLogProcessor, TrainingExample

logger = logging.getLogger(__name__)

@dataclass
class UnslothTrainingConfig:
    """Configuration for Unsloth fine-tuning."""
    # Model configuration
    model_name: str = "unsloth/llama-3-8b-bnb-4bit"
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    use_gradient_checkpointing: str = "unsloth"  # "unsloth", "transformers", or ""
    
    # Training configuration
    dataset_path: Optional[Union[str, Path]] = None
    output_dir: str = "models/finetuned"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    eval_accumulation_steps: Optional[int] = None
    optim: str = "adamw_8bit"
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 0.3
    
    # LoRA configuration
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    # Training monitoring
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: Optional[int] = None
    save_total_limit: int = 3
    evaluation_strategy: str = "steps"
    
    # Dataset configuration
    chat_template: Optional[str] = None  # Path to chat template or template string
    system_prompt: str = ""
    max_samples: Optional[int] = None
    
    # Distributed training
    local_rank: int = -1
    ddp_find_unused_parameters: bool = False
    fsdp: Optional[str] = None
    fsdp_config: Optional[Dict[str, Any]] = None
    
    # Other
    seed: int = 42
    fp16: bool = not torch.cuda.is_bf16_supported()
    bf16: bool = torch.cuda.is_bf16_supported()
    
    def __post_init__(self):
        """Validate configuration."""
        if self.load_in_4bit and self.load_in_8bit:
            raise ValueError("Cannot use both 4-bit and 8-bit quantization")
        if not (self.load_in_4bit or self.load_in_8bit):
            logger.warning("Neither 4-bit nor 8-bit quantization is enabled. This may require significant GPU memory.")


class UnslothFinetuner:
    """A class for fine-tuning models using Unsloth with support for chat logs."""
    
    def __init__(self, config: Optional[UnslothTrainingConfig] = None):
        """Initialize the Unsloth fine-tuner."""
        self.config = config or UnslothTrainingConfig()
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.processor = ChatLogProcessor(system_prompt=self.config.system_prompt)
        self._is_initialized = False
        
        # Set seed for reproducibility
        set_seed(self.config.seed)
        
    def _setup_distributed(self):
        """Set up distributed training if needed."""
        if self.config.local_rank != -1:
            torch.cuda.set_device(self.config.local_rank)
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=int(os.environ.get('WORLD_SIZE', 1)),
                rank=self.config.local_rank
            )
            logger.info(f"Initialized distributed training on rank {self.config.local_rank}")
    
    def load_model_and_tokenizer(self):
        """Load the model and tokenizer with Unsloth optimizations."""
        if self._is_initialized:
            return self.model, self.tokenizer
            
        logger.info(f"Loading model: {self.config.model_name}")
        
        # Determine compute dtype
        compute_dtype = None
        if self.config.bf16:
            compute_dtype = torch.bfloat16
        elif self.config.fp16:
            compute_dtype = torch.float16
            
        # Load model with Unsloth optimizations
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model_name,
            max_seq_length=self.config.max_seq_length,
            dtype=compute_dtype,
            load_in_4bit=self.config.load_in_4bit,
            load_in_8bit=self.config.load_in_8bit,
            token=os.environ.get("HF_TOKEN", None),  # For private models
        )
        
        # Set up LoRA if enabled
        if self.config.use_lora:
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=self.config.lora_rank,
                target_modules=self.config.lora_target_modules,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                use_gradient_checkpointing=self.config.use_gradient_checkpointing,
                random_state=self.config.seed,
            )
        
        self._is_initialized = True
        return self.model, self.tokenizer
    
    def prepare_dataset(self, dataset_path: Optional[Union[str, Path]] = None) -> Dataset:
        """Prepare the dataset for training."""
        dataset_path = dataset_path or self.config.dataset_path
        if not dataset_path:
            raise ValueError("No dataset path provided")
            
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
        # Load and process the dataset
        if dataset_path.suffix == '.json' or dataset_path.suffix == '.jsonl':
            # Process chat logs into training examples
            examples = self.processor.process_chat_logs(dataset_path)
            
            # Convert to Alpaca format for fine-tuning
            formatted_examples = self.processor.convert_to_alpaca_format(examples)
            
            # Create a dataset
            dataset = Dataset.from_list(formatted_examples)
        else:
            # Try to load using datasets library
            data_files = {"train": str(dataset_path)}
            dataset = load_dataset('json', data_files=data_files, split='train')
        
        # Apply max_samples if specified
        if self.config.max_samples is not None and len(dataset) > self.config.max_samples:
            dataset = dataset.select(range(self.config.max_samples))
        
        # Tokenize the dataset
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=self.config.max_seq_length,
                return_tensors="pt",
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
        )
        
        return tokenized_dataset
    
    def train(
        self,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        **training_kwargs
    ) -> None:
        """Train the model."""
        # Load model and tokenizer if not already loaded
        self.load_model_and_tokenizer()
        
        # Prepare datasets if not provided
        if train_dataset is None:
            train_dataset = self.prepare_dataset()
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            eval_accumulation_steps=self.config.eval_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler_type,
            max_grad_norm=self.config.max_grad_norm,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            save_total_limit=self.config.save_total_limit,
            evaluation_strategy=self.config.evaluation_strategy,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            ddp_find_unused_parameters=self.config.ddp_find_unused_parameters,
            fsdp=self.config.fsdp,
            fsdp_config=self.config.fsdp_config,
            local_rank=self.config.local_rank,
            remove_unused_columns=False,  # Important for custom datasets
            **training_kwargs
        )
        
        # Initialize trainer
        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field="text",
            max_seq_length=self.config.max_seq_length,
            tokenizer=self.tokenizer,
        )
        
        # Start training
        logger.info("Starting training...")
        self.trainer.train()
        
        # Save the final model
        self.trainer.save_model(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)
        logger.info(f"Training complete. Model saved to {self.config.output_dir}")
    
    def save_model(self, output_dir: Optional[Union[str, Path]] = None):
        """Save the model and tokenizer."""
        if not self._is_initialized:
            raise RuntimeError("Model not loaded. Call load_model_and_tokenizer() first.")
            
        output_dir = output_dir or self.config.output_dir
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the model
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(output_dir)
        
        # Save the tokenizer
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        
        # Save the config
        config_path = output_dir / "training_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        logger.info(f"Model saved to {output_dir}")


def train_with_chat_logs(
    chat_log_path: Union[str, Path],
    output_dir: Union[str, Path],
    model_name: str = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length: int = 2048,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 2,
    learning_rate: float = 2e-4,
    **kwargs
) -> UnslothFinetuner:
    """Convenience function to train a model with chat logs.
    
    Args:
        chat_log_path: Path to the chat log file (JSON or JSONL)
        output_dir: Directory to save the trained model
        model_name: Name of the base model to fine-tune
        max_seq_length: Maximum sequence length
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per device
        learning_rate: Learning rate
        **kwargs: Additional arguments to pass to UnslothTrainingConfig
        
    Returns:
        UnslothFinetuner: The trained model
    """
    # Set up config
    config = UnslothTrainingConfig(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dataset_path=str(chat_log_path),
        output_dir=str(output_dir),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        **kwargs
    )
    
    # Initialize and train
    finetuner = UnslothFinetuner(config)
    finetuner.train()
    
    return finetuner
