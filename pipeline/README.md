# Training Pipeline

This module provides a high-level interface for training and deploying domain expert models.

## Features

- **Easy-to-use API**: Simple interface for common workflows
- **Flexible Configuration**: Support for both simple and advanced use cases
- **Progress Tracking**: Built-in callbacks for monitoring training progress
- **Distributed Training**: Support for multi-GPU training
- **Model Deployment**: Easy deployment of trained models

## Quick Start

### Installation

```bash
# Install the required dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
import asyncio
from bitnet_sme_enhanced.pipeline import DomainExpert

async def main():
    # Create a new domain expert
    expert = DomainExpert.create(
        domain="quantum_physics",
        num_training_examples=500,
        num_epochs=2,
        output_dir="my_experts"
    )
    
    # Add a progress callback
    def log_progress(update):
        print(f"[{update.get('stage', 'unknown')}] {update.get('status', '')}")
    
    expert.add_callback(log_progress)
    
    # Train the model
    results = await expert.train()
    
    # Generate text
    response = await expert.generate("Explain quantum entanglement in simple terms.")
    print(response)
    
    # Deploy the model
    deployment = await expert.deploy()
    print(f"Model deployed at {deployment['api_url']}")
    
    # Save the configuration
    expert.save_config("quantum_physics_expert_config.yaml")

if __name__ == "__main__":
    asyncio.run(main())
```

## Command Line Interface

You can also use the provided example script to train models from the command line:

```bash
python examples/train_domain_expert.py --domain "quantum_physics" --examples 500 --epochs 2 --deploy
```

## Advanced Configuration

For advanced use cases, you can customize the training process by providing a configuration dictionary or file:

```python
config = {
    'domain': 'quantum_physics',
    'base_model': 'unsloth/llama-3-8b-bnb-4bit',
    'output_dir': 'my_experts',
    'data': {
        'num_examples': 1000,
        'max_seq_length': 2048
    },
    'training': {
        'num_epochs': 3,
        'batch_size': 4,
        'learning_rate': 2e-4,
        'lora_rank': 16,
        'lora_alpha': 32,
        'lora_dropout': 0.05
    },
    'generation': {
        'temperature': 0.7,
        'max_length': 200
    },
    'deployment': {
        'target': 'local',
        'api_port': 8000
    }
}

expert = DomainExpert(config)
```

## Callbacks

You can monitor the training progress by adding callbacks:

```python
def log_progress(update):
    stage = update.get('stage', 'unknown')
    status = update.get('status', '')
    message = update.get('message', '')
    
    if 'progress' in update:
        progress = update['progress']
        print(f"{stage.upper()}: {status} - {progress:.1f}% {message}")
    else:
        print(f"{stage.upper()}: {status} {message}")

expert.add_callback(log_progress)
```

## Model Deployment

The pipeline supports multiple deployment targets. Currently, the following targets are supported:

- `local`: Deploy a local FastAPI server
- (More deployment targets coming soon)

## Saving and Loading

You can save and load the expert configuration for later use:

```python
# Save the configuration
expert.save_config("my_expert_config.yaml")

# Later, load the expert
from bitnet_sme_enhanced.pipeline import DomainExpert

expert = DomainExpert.load("my_expert_config.yaml")
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
