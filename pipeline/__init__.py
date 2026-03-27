"""
High-level interface for the training pipeline.

This module provides a simple, user-friendly API for training and deploying
domain expert models.
"""
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Callable, Awaitable
import asyncio
import logging

from ..config.pipeline_config import PipelineConfig
from .integrated_training_pipeline import IntegratedTrainingPipeline, TrainingProgressCallback

logger = logging.getLogger(__name__)

class DomainExpert:
    """High-level interface for training and using domain expert models.
    
    This class provides a simple API for common workflows while maintaining
    access to advanced configuration options when needed.
    """
    
    def __init__(self, config: Optional[Union[PipelineConfig, Dict[str, Any]]] = None):
        """Initialize the domain expert with a configuration.
        
        Args:
            config: Either a PipelineConfig object or a dictionary of configuration options.
                   If None, default configuration will be used.
        """
        if config is None:
            raise ValueError("A configuration is required. Use DomainExpert.create() or provide a config.")
            
        if isinstance(config, dict):
            self.config = PipelineConfig.from_dict(config)
        else:
            self.config = config
            
        self._pipeline = None
        self._callbacks = []
    
    @classmethod
    def create(
        cls,
        domain: str,
        base_model: str = "unsloth/llama-3-8b-bnb-4bit",
        num_training_examples: int = 1000,
        num_epochs: int = 3,
        output_dir: Union[str, Path] = "pipelines",
        **kwargs
    ) -> 'DomainExpert':
        """Create a new domain expert with sensible defaults.
        
        This is the recommended way to create a new domain expert.
        
        Args:
            domain: The domain of expertise (e.g., "quantum_physics", "legal_docs")
            base_model: Base model to fine-tune
            num_training_examples: Number of training examples to generate
            num_epochs: Number of training epochs
            output_dir: Base directory for pipeline outputs
            **kwargs: Additional configuration options
            
        Returns:
            A configured DomainExpert instance
        """
        config = {
            'domain': domain,
            'base_model': base_model,
            'output_dir': output_dir,
            'data': {
                'num_examples': num_training_examples
            },
            'training': {
                'num_epochs': num_epochs
            }
        }
        
        # Update with any additional kwargs
        for key, value in kwargs.items():
            if '.' in key:
                # Handle nested keys (e.g., 'training.learning_rate')
                section, subkey = key.split('.', 1)
                if section in config and isinstance(config[section], dict):
                    config[section][subkey] = value
                else:
                    config[section] = {subkey: value}
            else:
                config[key] = value
        
        return cls(config)
    
    @classmethod
    def from_config_file(cls, config_path: Union[str, Path]) -> 'DomainExpert':
        """Load a domain expert from a configuration file."""
        config = PipelineConfig.load(config_path)
        return cls(config)
    
    def add_callback(self, callback: TrainingProgressCallback):
        """Add a callback for training progress updates."""
        self._callbacks.append(callback)
    
    async def train(self, **kwargs) -> Dict[str, Any]:
        """Train the domain expert model.
        
        Args:
            **kwargs: Override any training parameters
            
        Returns:
            Dictionary with training results
        """
        # Update config with any overrides
        for key, value in kwargs.items():
            if hasattr(self.config.training, key):
                setattr(self.config.training, key, value)
            elif hasattr(self.config.data, key):
                setattr(self.config.data, key, value)
            else:
                setattr(self.config, key, value)
        
        # Create pipeline if it doesn't exist
        if self._pipeline is None:
            self._pipeline = IntegratedTrainingPipeline(
                domain=self.config.domain,
                base_model=self.config.base_model,
                output_dir=self.config.output_dir,
                local_rank=self.config.local_rank,
                world_size=self.world_size,
                seed=self.config.seed
            )
        
        # Add callbacks
        for callback in self._callbacks:
            self._pipeline.add_callback(callback)
        
        # Run training
        results = await self._pipeline.train(
            train_data_path=self.config.data.dataset_path,
            **self.config.training.__dict__
        )
        
        return results
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using the trained model.
        
        Args:
            prompt: Input text prompt
            **kwargs: Override generation parameters
            
        Returns:
            Generated text
        """
        if self._pipeline is None:
            raise ValueError("Model not trained. Call train() first.")
            
        # Update generation config with any overrides
        gen_config = self.config.generation.__dict__.copy()
        gen_config.update(kwargs)
        
        return await self._pipeline.generate(prompt, **gen_config)
    
    async def deploy(self, **kwargs) -> Dict[str, Any]:
        """Deploy the trained model.
        
        Args:
            **kwargs: Override deployment parameters
            
        Returns:
            Deployment information
        """
        if self._pipeline is None:
            raise ValueError("Model not trained. Call train() first.")
            
        # Update deployment config with any overrides
        deploy_config = self.config.deployment.__dict__.copy()
        deploy_config.update(kwargs)
        
        return await self._pipeline.deploy(
            model_path=self._pipeline.trained_model_path,
            **deploy_config
        )
    
    def save_config(self, path: Optional[Union[str, Path]] = None) -> None:
        """Save the current configuration to a file."""
        self.config.save(path)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'DomainExpert':
        """Load a domain expert from a saved configuration."""
        return cls.from_config_file(path)

# Example usage:
"""
# Create a new domain expert
expert = await DomainExpert.create(
    domain="quantum_physics",
    num_training_examples=500,
    num_epochs=2
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

# Later, load the expert
# expert = DomainExpert.load("quantum_physics_expert_config.yaml")
"""
