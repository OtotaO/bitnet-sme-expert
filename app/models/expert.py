from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Type, TypeVar, Generic
from pydantic import BaseModel, Field, validator
import logging
import time
from datetime import datetime
import uuid

from ..schemas.base import ExpertDomain

logger = logging.getLogger(__name__)

T = TypeVar('T')

class ExpertConfig(BaseModel):
    """Configuration for an expert."""
    name: str = Field(..., description="Name of the expert")
    description: str = Field(..., description="Description of the expert's capabilities")
    domain: ExpertDomain = Field(..., description="Domain of expertise")
    version: str = Field("1.0.0", description="Expert version")
    model_name: str = Field("gpt-3.5-turbo", description="Name of the underlying model")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    max_tokens: int = Field(512, ge=1, le=4096, description="Maximum number of tokens to generate")
    stop_sequences: List[str] = Field(
        default_factory=list,
        description="Stop sequences for generation"
    )
    is_custom: bool = Field(False, description="Whether this is a custom expert")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional configuration parameters"
    )

class ExpertContext(BaseModel):
    """Context for expert generation."""
    session_id: Optional[str] = Field(
        None,
        description="Session identifier for multi-turn conversations"
    )
    user_id: Optional[str] = Field(
        None,
        description="Identifier for the user making the request"
    )
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context for the expert"
    )

class BaseExpert(ABC):
    """Base class for all expert implementations."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the expert with configuration."""
        self.config = ExpertConfig(**(config or {}))
        self.initialized = False
        self.logger = logger.getChild(f"expert.{self.config.domain}.{self.config.name}")
        self.id = str(uuid.uuid4())
        
    @property
    def name(self) -> str:
        """Get the expert's name."""
        return self.config.name
        
    @property
    def domain(self) -> ExpertDomain:
        """Get the expert's domain."""
        return self.config.domain
        
    async def initialize(self):
        """Initialize the expert (lazy loading)."""
        if not self.initialized:
            start_time = time.time()
            self.logger.info(f"Initializing {self.__class__.__name__}")
            await self._initialize()
            self.initialized = True
            self.logger.info(
                f"Initialized {self.__class__.__name__} "
                f"in {(time.time() - start_time):.2f}s"
            )
            
    async def _initialize(self):
        """Subclass-specific initialization."""
        pass
        
    async def generate(
        self,
        input_text: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a response to the input text.
        
        Args:
            input_text: The input text to respond to
            context: Additional context for the expert
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary containing the expert's response and metadata
        """
        start_time = time.time()
        
        try:
            if not self.initialized:
                await self.initialize()
                
            # Update config with any overrides
            config = self.config.dict()
            config.update(kwargs)
            
            self.logger.debug(f"Generating response for input: {input_text[:200]}...")
            
            # Call the implementation
            raw_response = await self._generate_impl(input_text, context or {}, **config)

            response = self._normalize_expert_output(raw_response)

            # Add timing information
            processing_time = time.time() - start_time
            response["metadata"].update({
                "processing_time": processing_time,
                "model": self.config.model_name,
                "expert_id": self.id,
                "expert_name": self.config.name,
                "expert_domain": self.config.domain,
                "timestamp": datetime.utcnow().isoformat()
            })
            response["tokens_used"] = int(response.get("tokens_used", 0))
            response["metadata"]["tokens_used"] = response["tokens_used"]
            response["metadata"]["confidence"] = response.get("confidence", 0.0)
            response["metadata"]["sources_count"] = len(response.get("sources", []))
                
            self.logger.debug(
                f"Generated response in {processing_time:.2f}s: "
                f"{response.get('response', '')[:200]}..."
            )
            
            return response
            
        except Exception as e:
            self.logger.error(
                f"Error in {self.__class__.__name__}.generate: {str(e)}",
                exc_info=True
            )
            raise

    def _normalize_expert_output(self, output: Any) -> Dict[str, Any]:
        """Normalize expert output to a stable response schema."""
        if not isinstance(output, dict):
            output = {"response": str(output)}

        metadata = output.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}

        sources = output.get("sources")
        if not isinstance(sources, list):
            sources = []

        response_text = str(output.get("response", ""))
        tokens_used = output.get("tokens_used", metadata.get("tokens_used"))
        if tokens_used is None:
            tokens_used = len(response_text.split())

        normalized = {
            "response": response_text,
            "confidence": float(output.get("confidence", metadata.get("confidence", 0.0))),
            "tokens_used": int(tokens_used),
            "model": str(output.get("model", metadata.get("model", self.config.model_name))),
            "metadata": metadata,
            "sources": sources
        }
        return normalized
            
    @abstractmethod
    async def _generate_impl(
        self,
        input_text: str,
        context: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Subclass-specific implementation of generation.
        
        Args:
            input_text: The input text to respond to
            context: Additional context for the expert
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary containing the expert's response and metadata
        """
        pass
        
    async def cleanup(self):
        """Clean up any resources used by the expert."""
        if self.initialized:
            try:
                await self._cleanup()
                self.initialized = False
                self.logger.info(f"Cleaned up {self.__class__.__name__}")
            except Exception as e:
                self.logger.error(
                    f"Error cleaning up {self.__class__.__name__}: {str(e)}",
                    exc_info=True
                )
                
    async def _cleanup(self):
        """Subclass-specific cleanup."""
        pass
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the expert to a dictionary."""
        return {
            "id": self.id,
            "name": self.config.name,
            "description": self.config.description,
            "domain": self.config.domain,
            "version": self.config.version,
            "model": self.config.model_name,
            "is_custom": self.config.is_custom,
            "initialized": self.initialized,
            "metadata": self.config.metadata
        }
