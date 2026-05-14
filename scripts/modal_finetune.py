"""Fine-tune any HF base on Modal with Unsloth + TRL, then push to the HF Hub.

Usage:

    modal run scripts/modal_finetune.py::train \\
        --base-model meta-llama/Llama-3.3-70B-Instruct \\
        --dataset-repo your-org/your-domain-sft \\
        --output-repo your-org/llama-3.3-70b-domain-lora

What it does:

1. Spins a Modal H100 (or H100:2 / A100-80GB, configurable).
2. Pulls the base model into a persistent Modal Volume (cached across runs).
3. Runs LoRA SFT with Unsloth + TRL ``SFTConfig`` — same config dataclass the
   API exposes via ``app/services/fine_tuning.py``.
4. Pushes the resulting LoRA adapter to a HF repo of your choice.

The dataset is expected to be a HF dataset with either a ``text`` column or
``instruction`` + ``response`` columns (matching ``FineTuningService.load_dataset``).

This is the Modal-substrate sibling of the local-machine path in
``app/services/fine_tuning.py``. The two share the ``TrainingConfig`` dataclass
so the same knobs work whether you train on a workstation or on Modal.
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    import modal
except ImportError as exc:  # pragma: no cover - script-time only
    raise SystemExit("modal is not installed. `pip install modal` then `modal setup`.") from exc


GPU = os.environ.get("GPU", "H100")
TIMEOUT_S = int(os.environ.get("TIMEOUT_S", str(4 * 60 * 60)))  # 4 hours by default

app = modal.App("dspy-sme-finetune")

WEIGHTS_VOLUME = modal.Volume.from_name("dspy-sme-finetune-cache", create_if_missing=True)
HF_CACHE = "/root/.cache/huggingface"
OUTPUTS = "/outputs"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "unsloth>=2025.10",
        "transformers>=4.46",
        "trl>=0.12",
        "peft>=0.13",
        "accelerate>=1.0",
        "datasets>=3.0",
        "huggingface_hub[hf_transfer]>=0.27",
        "torch>=2.5",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


@app.function(
    image=image,
    gpu=GPU,
    volumes={
        HF_CACHE: WEIGHTS_VOLUME,
        OUTPUTS: modal.Volume.from_name("dspy-sme-finetune-outputs", create_if_missing=True),
    },
    secrets=[modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])],
    timeout=TIMEOUT_S,
)
def train(
    base_model: str = "unsloth/llama-3.1-8b-bnb-4bit",
    dataset_repo: str = "",
    output_repo: str = "",
    max_seq_length: int = 2048,
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    lora_rank: int = 16,
    lora_alpha: int = 16,
    seed: int = 42,
) -> str:
    """Run LoRA SFT on Modal and (optionally) push to the HF Hub.

    Returns the local output path inside the Modal Volume. If ``output_repo``
    is set, the adapter is also pushed to that HF repo (creates if missing).
    """
    import torch
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

    if not dataset_repo:
        raise ValueError("--dataset-repo is required")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"[modal_finetune] loading {base_model} on {GPU} ({dtype})")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=True,
        token=os.environ.get("HF_TOKEN"),
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
    )

    print(f"[modal_finetune] loading dataset {dataset_repo}")
    ds = load_dataset(dataset_repo, split="train")
    if "text" not in ds.column_names and {"instruction", "response"}.issubset(ds.column_names):
        ds = ds.map(
            lambda ex: {
                "text": f"### Instruction:\n{ex['instruction']}\n\n### Response:\n{ex['response']}"
            },
            remove_columns=[c for c in ds.column_names if c != "text"],
        )

    out_dir = f"{OUTPUTS}/{base_model.replace('/', '_')}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.0,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        max_grad_norm=0.3,
        optim="adamw_8bit",
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        seed=seed,
        max_length=max_seq_length,
        dataset_text_field="text",
        packing=True,
        report_to="none",
    )

    trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=ds, args=sft_config)
    print("[modal_finetune] training")
    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[modal_finetune] saved adapter to {out_dir}")

    if output_repo:
        print(f"[modal_finetune] pushing adapter to {output_repo}")
        api = HfApi(token=os.environ["HF_TOKEN"])
        api.create_repo(repo_id=output_repo, exist_ok=True, private=True)
        api.upload_folder(folder_path=out_dir, repo_id=output_repo)
        print(f"[modal_finetune] done — adapter at https://huggingface.co/{output_repo}")

    return out_dir


@app.local_entrypoint()
def main(
    base_model: str = "unsloth/llama-3.1-8b-bnb-4bit",
    dataset_repo: str = "",
    output_repo: str = "",
) -> None:
    """Local entrypoint so ``modal run scripts/modal_finetune.py`` works.

    Forward CLI args to the remote ``train`` function.
    """
    if not dataset_repo:
        raise SystemExit(
            "usage: modal run scripts/modal_finetune.py "
            "--base-model <hf-id> --dataset-repo <hf-id> [--output-repo <hf-id>]"
        )
    train.remote(
        base_model=base_model,
        dataset_repo=dataset_repo,
        output_repo=output_repo,
    )
