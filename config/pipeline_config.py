"""Configuration for the training pipeline."""
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

@dataclass
class DataConfig:
    """Configuration for data generation and loading."""
    num_examples: int = 1000
    seed_examples: int = 10
    max_seq_length: int = 2048
    train_ratio: float = 0.9
    overwrite: bool = False
    dataset_path: Optional[Path] = None

@dataclass
class TrainingConfig:
    """Configuration for model training."""
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 0.3
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    save_steps: int = 100
    logging_steps: int = 10
    eval_steps: Optional[int] = None
    output_dir: Optional[Path] = None

@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_length: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.2

@dataclass
class DeploymentConfig:
    """Configuration for model deployment."""
    target: str = "local"  # local, sagemaker, vertex-ai, etc.
    api_port: int = 8000
    instance_type: Optional[str] = None
    endpoint_name: Optional[str] = None

@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""
    domain: str
    base_model: str = "unsloth/llama-3-8b-bnb-4bit"
    output_dir: Path = field(default_factory=lambda: Path("pipelines"))
    seed: int = 42
    local_rank: int = -1
    world_size: int = 1
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)
    
    def __post_init__(self):
        """Set up paths and validate configuration."""
        if not isinstance(self.output_dir, Path):
            self.output_dir = Path(self.output_dir)
            
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = self.output_dir / f"{self.domain}_{timestamp}"
        
        # Set default output dir for training if not specified
        if self.training.output_dir is None:
            self.training.output_dir = self.output_dir / "model"
            
        # Set default dataset path if not specified
        if self.data.dataset_path is None:
            self.data.dataset_path = self.output_dir / "data" / "train.jsonl"

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PipelineConfig':
        """Create config from dictionary with nested structure support."""
        # Create copies to avoid modifying the input
        config_dict = config_dict.copy()
        
        # Handle nested configs
        data_config = DataConfig(**config_dict.pop('data', {}))
        training_config = TrainingConfig(**config_dict.pop('training', {}))
        gen_config = GenerationConfig(**config_dict.pop('generation', {}))
        deploy_config = DeploymentConfig(**config_dict.pop('deployment', {}))
        
        return cls(
            **config_dict,
            data=data_config,
            training=training_config,
            generation=gen_config,
            deployment=deploy_config
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'domain': self.domain,
            'base_model': self.base_model,
            'output_dir': str(self.output_dir),
            'seed': self.seed,
            'local_rank': self.local_rank,
            'world_size': self.world_size,
            'data': {k: v for k, v in self.data.__dict__.items() 
                    if not k.startswith('_') and k != 'dataset_path'},
            'training': {k: v for k, v in self.training.__dict__.items() 
                       if not k.startswith('_') and k != 'output_dir'},
            'generation': self.generation.__dict__,
            'deployment': self.deployment.__dict__
        }
    
    def save(self, path: Optional[Union[str, Path]] = None) -> None:
        """Save config to file."""
        if path is None:
            path = self.output_dir / "config.yaml"
        elif not isinstance(path, Path):
            path = Path(path)
            
        path.parent.mkdir(parents=True, exist_ok=True)
        import yaml
        with open(path, 'w') as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'PipelineConfig':
        """Load config from file."""
        if not isinstance(path, Path):
            path = Path(path)
            
        import yaml
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
