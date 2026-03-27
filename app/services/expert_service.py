import logging
import importlib
import asyncio
from typing import Dict, Type, List, Optional, Any, TypeVar, Generic, Union
from datetime import datetime
import uuid

from ..models.expert import BaseExpert, ExpertConfig, ExpertContext
from ..schemas.base import ExpertDomain, BaseResponse
from ..schemas.response import ExpertInfo, ListExpertsResponse
from ..schemas.request import QueryRequest, CollaborateRequest

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseExpert)

class ExpertService:
    """Service for managing and interacting with experts."""
    
    def __init__(self):
        """Initialize the expert service."""
        self._experts: Dict[str, BaseExpert] = {}
        self._expert_classes: Dict[ExpertDomain, Type[BaseExpert]] = {}
        self._expert_configs: Dict[str, Dict[str, Any]] = {}
        self._initialized = False
        self.logger = logger.getChild("ExpertService")
    
    async def initialize(self):
        """Initialize the expert service and all registered experts."""
        if self._initialized:
            return
            
        self.logger.info("Initializing ExpertService")
        start_time = datetime.now()
        
        # Initialize all registered experts
        init_tasks = []
        for expert_id, expert in self._experts.items():
            if not expert.initialized:
                init_tasks.append(expert.initialize())
        
        if init_tasks:
            await asyncio.gather(*init_tasks, return_exceptions=True)
        
        self._initialized = True
        self.logger.info(
            f"ExpertService initialized with {len(self._experts)} experts "
            f"in {(datetime.now() - start_time).total_seconds():.2f}s"
        )
    
    def register_expert_class(
        self,
        domain: ExpertDomain,
        expert_class: Type[BaseExpert],
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register an expert class for a specific domain.
        
        Args:
            domain: The domain of expertise
            expert_class: The expert class to register
            config: Optional configuration for the expert
        """
        if not issubclass(expert_class, BaseExpert):
            raise ValueError(f"Expert class must be a subclass of BaseExpert")
            
        self._expert_classes[domain] = expert_class
        
        # Create a default config if none provided
        if config is None:
            config = {
                "name": f"{domain.value}_expert",
                "description": f"{domain.value.capitalize()} expert",
                "domain": domain,
            }
            
        self._expert_configs[domain.value] = config
        self.logger.info(f"Registered expert class {expert_class.__name__} for domain {domain}")
    
    async def create_expert(
        self,
        domain: Union[ExpertDomain, str],
        config: Optional[Dict[str, Any]] = None,
        expert_id: Optional[str] = None
    ) -> BaseExpert:
        """
        Create a new expert instance.
        
        Args:
            domain: The domain of expertise
            config: Configuration for the expert
            expert_id: Optional custom ID for the expert
            
        Returns:
            The created expert instance
        """
        if isinstance(domain, str):
            try:
                domain = ExpertDomain(domain.lower())
            except ValueError as e:
                raise ValueError(f"Invalid expert domain: {domain}") from e
        
        if domain not in self._expert_classes:
            raise ValueError(f"No expert class registered for domain: {domain}")
        
        # Merge with default config
        default_config = self._expert_configs.get(domain.value, {})
        if config:
            default_config.update(config)
        
        # Create the expert
        expert_class = self._expert_classes[domain]
        expert = expert_class(default_config)
        
        # Set custom ID if provided
        if expert_id:
            # This requires adding an id attribute to BaseExpert
            expert.id = expert_id
            
        # Initialize the expert if the service is already initialized
        if self._initialized:
            await expert.initialize()
            
        # Store the expert
        self._experts[expert.id] = expert
        self.logger.info(f"Created expert {expert.id} for domain {domain}")
        
        return expert
    
    async def get_expert(
        self,
        expert_id: str,
        domain: Optional[Union[ExpertDomain, str]] = None
    ) -> BaseExpert:
        """
        Get an expert by ID and optionally verify its domain.
        
        Args:
            expert_id: The ID of the expert to get
            domain: Optional domain to verify the expert against
            
        Returns:
            The expert instance
            
        Raises:
            ValueError: If the expert is not found or domain doesn't match
        """
        expert = self._experts.get(expert_id)
        if not expert:
            raise ValueError(f"Expert not found: {expert_id}")
            
        if domain is not None:
            if isinstance(domain, str):
                domain = ExpertDomain(domain.lower())
                
            if expert.domain != domain:
                raise ValueError(
                    f"Expert {expert_id} is a {expert.domain.value} expert, "
                    f"not a {domain.value} expert"
                )
                
        return expert
    
    async def get_experts_by_domain(
        self,
        domain: Union[ExpertDomain, str],
        only_initialized: bool = True
    ) -> List[BaseExpert]:
        """
        Get all experts for a specific domain.
        
        Args:
            domain: The domain of expertise
            only_initialized: Whether to only return initialized experts
            
        Returns:
            List of expert instances
        """
        if isinstance(domain, str):
            domain = ExpertDomain(domain.lower())
            
        experts = [
            expert for expert in self._experts.values()
            if expert.domain == domain and (not only_initialized or expert.initialized)
        ]
        
        return experts
    
    async def get_all_experts(self, only_initialized: bool = True) -> Dict[str, BaseExpert]:
        """
        Get all experts.
        
        Args:
            only_initialized: Whether to only return initialized experts
            
        Returns:
            Dictionary mapping expert IDs to expert instances
        """
        if only_initialized:
            return {
                expert_id: expert
                for expert_id, expert in self._experts.items()
                if expert.initialized
            }
        return dict(self._experts)
    
    async def query_expert(
        self,
        expert_id: str,
        request: QueryRequest,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query a specific expert.
        
        Args:
            expert_id: The ID of the expert to query
            request: The query request
            context: Additional context for the expert
            
        Returns:
            The expert's response
        """
        expert = await self.get_expert(expert_id)
        
        # Prepare the context
        expert_context = {
            "question": request.question,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "context": context or {}
        }
        
        # Generate the response
        response = await expert.generate(
            input_text=request.question,
            context=expert_context,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        return response
    
    async def collaborate(
        self,
        request: CollaborateRequest,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Collaborate with multiple experts.
        
        Args:
            request: The collaboration request
            context: Additional context for the experts
            
        Returns:
            Dictionary mapping expert IDs to their responses
        """
        # Get all experts if no specific domains are provided
        if not request.domains:
            experts = await self.get_all_experts()
        else:
            # Get experts for the specified domains
            experts = {}
            for domain in request.domains:
                domain_experts = await self.get_experts_by_domain(domain)
                for expert in domain_experts:
                    experts[expert.id] = expert
        
        if not experts:
            raise ValueError("No experts available for the specified domains")
        
        # Prepare the context
        expert_context = {
            "question": request.question,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "context": context or {}
        }
        
        # Query all experts in parallel
        tasks = []
        for expert_id, expert in experts.items():
            task = asyncio.create_task(
                self._query_expert_safe(
                    expert,
                    request.question,
                    expert_context,
                    request.max_tokens,
                    request.temperature,
                    request.top_p
                )
            )
            tasks.append((expert_id, task))
        
        # Wait for all tasks to complete
        results = {}
        for expert_id, task in tasks:
            try:
                result = await task
                results[expert_id] = result
            except Exception as e:
                self.logger.error(f"Error querying expert {expert_id}: {str(e)}", exc_info=True)
                results[expert_id] = {
                    "error": str(e),
                    "success": False
                }
        
        return results
    
    async def _query_expert_safe(
        self,
        expert: BaseExpert,
        question: str,
        context: Dict[str, Any],
        max_tokens: int,
        temperature: float,
        top_p: float
    ) -> Dict[str, Any]:
        """Safely query an expert with error handling."""
        try:
            response = await expert.generate(
                input_text=question,
                context=context,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            response["success"] = True
            return response
        except Exception as e:
            self.logger.error(
                f"Error in expert {expert.id} ({expert.domain}): {str(e)}",
                exc_info=True
            )
            return {
                "error": str(e),
                "success": False,
                "expert_id": expert.id,
                "expert_domain": expert.domain.value
            }
    
    async def list_experts(self) -> List[Dict[str, Any]]:
        """
        List all available experts with their information.
        
        Returns:
            List of expert information dictionaries
        """
        experts = await self.get_all_experts()
        return [
            {
                "id": expert.id,
                "name": expert.name,
                "domain": expert.domain.value,
                "description": getattr(expert.config, "description", ""),
                "model": getattr(expert.config, "model_name", "unknown"),
                "initialized": expert.initialized,
                "is_custom": getattr(expert.config, "is_custom", False)
            }
            for expert in experts.values()
        ]
    
    async def cleanup(self):
        """Clean up all experts and release resources."""
        self.logger.info("Cleaning up ExpertService and all experts")
        cleanup_tasks = [expert.cleanup() for expert in self._experts.values()]
        await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        self._experts.clear()
        self._initialized = False
        self.logger.info("ExpertService cleanup complete")
