"""Supervised fine-tuning with Unsloth + TRL.

This module wraps the current (2026) Unsloth + TRL API for LoRA/QLoRA SFT.
Heavy dependencies live in the ``[finetune]`` extra; import-failure is handled
lazily so the main DSPy service starts even when these aren't installed.

Highlights of the modernized pipeline:

* Uses ``trl.SFTConfig`` (introduced post-TRL 0.9) instead of bare
  ``transformers.TrainingArguments``. ``TrainingArguments`` is still accepted
  for back-compat, but SFTConfig surfaces SFT-specific options cleanly.
* ``eval_strategy`` (was ``evaluation_strategy``, deprecated in transformers 4.46).
* ``torch.cuda.is_bf16_supported()`` instead of the removed
  ``unsloth.is_bfloat16_supported``.
* Tokenization is delegated to ``SFTTrainer`` — we pass raw text and a
  ``dataset_text_field``, which works against the current TRL signature where
  pre-tokenized vs. raw is auto-detected.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for an SFT run."""

    model_name: str = "unsloth/llama-3.1-8b-bnb-4bit"
    dataset_path: str | None = None
    output_dir: str = "models/finetuned"

    # Sequence shape
    max_seq_length: int = 2048

    # Optimizer / schedule
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 0.3
    optim: str = "adamw_8bit"

    # Logging / checkpoints
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int | None = None
    eval_strategy: str = "no"
    save_total_limit: int = 3

    # LoRA
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )

    # Quantization
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    use_gradient_checkpointing: str | bool = "unsloth"

    # Misc
    seed: int = 42
    chat_template: str | None = None
    system_prompt: str = ""
    max_samples: int | None = None

    # Distributed
    local_rank: int = -1

    def __post_init__(self) -> None:
        if self.load_in_4bit and self.load_in_8bit:
            raise ValueError("Cannot enable both 4-bit and 8-bit quantization")


class FineTuningService:
    """SFT service using Unsloth + TRL.

    All heavy imports happen inside methods so ``app/main.py`` doesn't fail
    when the ``[finetune]`` extra isn't installed.
    """

    def __init__(self, config: TrainingConfig | None = None) -> None:
        self.config = config or TrainingConfig()
        self.model: Any = None
        self.tokenizer: Any = None
        self.trainer: Any = None
        self._initialized = False

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self) -> tuple[Any, Any]:
        if self._initialized:
            return self.model, self.tokenizer

        import torch
        from unsloth import FastLanguageModel

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        logger.info(
            "finetune.load_model", extra={"model": self.config.model_name, "dtype": str(dtype)}
        )

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model_name,
            max_seq_length=self.config.max_seq_length,
            dtype=dtype,
            load_in_4bit=self.config.load_in_4bit,
            load_in_8bit=self.config.load_in_8bit,
            token=os.environ.get("HF_TOKEN"),
        )

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

        self._initialized = True
        return self.model, self.tokenizer

    # ------------------------------------------------------------------
    # Dataset prep
    # ------------------------------------------------------------------

    def load_dataset(self) -> Any:
        from datasets import Dataset, load_dataset

        path = self.config.dataset_path
        if not path:
            raise ValueError("dataset_path is required")
        if Path(path).suffix in (".jsonl", ".json"):
            ds = Dataset.from_json(path)
        else:
            ds = load_dataset(path, split="train")
        if self.config.max_samples and len(ds) > self.config.max_samples:
            ds = ds.select(range(self.config.max_samples))

        # Standardize to a 'text' field. Accepts either pre-rendered 'text' or
        # an Alpaca-like 'instruction'/'response' shape.
        cols = ds.column_names
        if "text" in cols:
            return ds
        if {"instruction", "response"}.issubset(cols):
            return ds.map(
                lambda ex: {"text": f"### Instruction:\n{ex['instruction']}\n\n### Response:\n{ex['response']}"},
                remove_columns=[c for c in cols if c != "text"],
            )
        raise ValueError(f"Don't know how to format dataset with columns {cols}")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self) -> None:
        from trl import SFTConfig, SFTTrainer

        self.load_model()
        dataset = self.load_dataset()

        sft_config = SFTConfig(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler_type,
            max_grad_norm=self.config.max_grad_norm,
            optim=self.config.optim,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            eval_strategy=self.config.eval_strategy,
            save_total_limit=self.config.save_total_limit,
            seed=self.config.seed,
            local_rank=self.config.local_rank,
            max_length=self.config.max_seq_length,
            dataset_text_field="text",
            packing=True,
            report_to="none",
        )

        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            args=sft_config,
        )
        logger.info("finetune.train.start", extra={"output_dir": self.config.output_dir})
        self.trainer.train()
        self.save_model()
        logger.info("finetune.train.done")

    def save_model(self, output_dir: str | None = None) -> None:
        target = output_dir or self.config.output_dir
        Path(target).mkdir(parents=True, exist_ok=True)
        if self.trainer is not None:
            self.trainer.save_model(target)
        elif self.model is not None:
            self.model.save_pretrained(target)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(target)
        (Path(target) / "training_config.json").write_text(
            json.dumps(self.config.__dict__, indent=2, default=str)
        )

    @classmethod
    def from_pretrained(cls, model_path: str) -> "FineTuningService":
        from unsloth import FastLanguageModel

        cfg_path = Path(model_path) / "training_config.json"
        cfg_data = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        cfg = TrainingConfig(**{k: v for k, v in cfg_data.items() if k in TrainingConfig.__dataclass_fields__})
        cfg.output_dir = model_path
        instance = cls(cfg)
        instance.model, instance.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=cfg.max_seq_length,
            load_in_4bit=cfg.load_in_4bit,
            load_in_8bit=cfg.load_in_8bit,
        )
        instance._initialized = True
        return instance
