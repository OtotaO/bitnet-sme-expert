"""Smoke tests for the Modal scripts.

These don't actually connect to Modal — they just verify the scripts parse,
expose the expected app names + entrypoints, and pick up env-var defaults
correctly. The full deployment is exercised by the user with `modal deploy`
/ `modal run`, gated by their Modal credentials.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load(name: str) -> object:
    """Import a script as a module without executing __main__."""
    pytest.importorskip("modal")
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_modal_serve_exposes_named_app(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
    mod = _load("modal_serve")
    assert hasattr(mod, "app"), "modal_serve must define `app = modal.App(...)`"
    assert mod.MODEL_NAME == "Qwen/Qwen2.5-7B-Instruct"
    assert mod.VLLM_PORT == 8000
    # The serve function should exist and be callable through Modal's decorator.
    assert hasattr(mod, "serve")


def test_modal_serve_honors_gpu_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GPU", "A10G")
    # The module reads env at import time; need to drop any cached copy.
    sys.modules.pop("modal_serve", None)
    mod = _load("modal_serve")
    assert mod.GPU == "A10G"


def test_modal_finetune_has_train_function() -> None:
    mod = _load("modal_finetune")
    assert hasattr(mod, "app"), "modal_finetune must define a Modal app"
    assert hasattr(mod, "train"), "modal_finetune must define train()"
    # local_entrypoint registers `main` for `modal run scripts/modal_finetune.py`
    assert hasattr(mod, "main")


def test_modal_scripts_use_persistent_volumes() -> None:
    """Both scripts must use named volumes so HF weights survive cold starts.

    Spot-check via source so we don't have to instantiate Modal objects.
    """
    serve_src = (SCRIPTS / "modal_serve.py").read_text()
    finetune_src = (SCRIPTS / "modal_finetune.py").read_text()
    assert "Volume.from_name" in serve_src, "modal_serve must cache HF weights"
    assert "Volume.from_name" in finetune_src, "modal_finetune must cache HF weights"
    assert "huggingface" in serve_src, "modal_serve must wire the huggingface Secret"
    assert "huggingface" in finetune_src, "modal_finetune must wire the huggingface Secret"


def test_modal_scripts_documented_in_env_example() -> None:
    """The substrate playbook references env-var patterns; .env.example
    should document at least the Modal API base + key pattern."""
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env.example")
    with open(env_path, encoding="utf-8") as f:
        content = f.read()
    assert "modal" in content.lower(), "Modal substrate must be documented in .env.example"
    assert "huggingface/auto/" in content, "HF Inference Providers usage must be documented"
