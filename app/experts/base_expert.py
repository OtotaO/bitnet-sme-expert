"""Base expert module with common functionality for all experts."""

from typing import Dict, Any, Optional, List, Union
import json
import re
import random
from datetime import datetime

from ..models.expert import BaseExpert, ExpertConfig
from ..schemas.base import ExpertDomain


class BaseExpertImpl(BaseExpert):
    """Base implementation for all expert types."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the base expert.
        
        Args:
            config: Configuration dictionary for the expert
        """
        # Set default config if not provided
        if config is None:
            config = {}
            
        # Ensure required config values are set
        config.setdefault("name", self.__class__.__name__)
        config.setdefault("description", f"{self.__class__.__name__} expert")
        config.setdefault("domain", self.DOMAIN.value)
        
        super().__init__(config)
    
    async def _initialize(self):
        """Initialize the expert with any required setup."""
        self.logger.info(f"Initializing {self.__class__.__name__}")
        
        # Load any required models or resources here
        self.initialized = True
        
    async def _generate_impl(
        self,
        input_text: str,
        context: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Generate a response to the input text.
        
        Args:
            input_text: The input text to respond to
            context: Additional context for the expert
            **kwargs: Additional parameters for generation
            
        Returns:
            Dictionary containing the response and metadata
        """
        raise NotImplementedError("Subclasses must implement _generate_impl")
    
    async def _cleanup(self):
        """Clean up any resources used by the expert."""
        self.logger.info(f"Cleaning up {self.__class__.__name__}")
        # Release any resources here
        
    def _extract_code_blocks(self, text: str) -> List[Dict[str, str]]:
        """Extract code blocks from markdown text.
        
        Args:
            text: The text to extract code blocks from
            
        Returns:
            List of dictionaries with 'language' and 'code' keys
        """
        pattern = r"```(\w*)\n([\s\S]*?)\n```"
        matches = re.finditer(pattern, text)
        
        blocks = []
        for match in matches:
            language = match.group(1) or "text"
            code = match.group(2).strip()
            blocks.append({"language": language, "code": code})
            
        return blocks
    
    def _format_response(
        self,
        response: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Format the expert's response with metadata.
        
        Args:
            response: The generated response text
            sources: List of sources/references used
            metadata: Additional metadata to include
            **kwargs: Additional fields to include
            
        Returns:
            Formatted response dictionary
        """
        if sources is None:
            sources = []
            
        if metadata is None:
            metadata = {}
            
        # Extract code blocks if present
        code_blocks = self._extract_code_blocks(response)
        
        return {
            "response": response,
            "sources": sources,
            "metadata": {
                "expert": self.config.name,
                "domain": self.config.domain,
                "model": self.config.model_name,
                "timestamp": datetime.utcnow().isoformat(),
                "code_blocks": code_blocks,
                **metadata
            },
            **kwargs
        }
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the expert.
        
        Returns:
            The system prompt string
        """
        return (
            f"You are a {self.config.name}, an AI expert in {self.config.domain}.\n"
            f"{self.config.description}\n"
            "Provide clear, accurate, and helpful responses to the user's questions.\n"
            "If you're not sure about something, say so rather than making things up."
        )
    
    def _get_examples(self) -> List[Dict[str, str]]:
        """Get example inputs and outputs for few-shot learning.
        
        Returns:
            List of example dictionaries with 'input' and 'output' keys
        """
        return []
    
    def _should_use_tool(self, input_text: str) -> bool:
        """Determine if a tool should be used for the given input.
        
        Args:
            input_text: The input text to analyze
            
        Returns:
            True if a tool should be used, False otherwise
        """
        # Default implementation - can be overridden by subclasses
        return False
    
    async def _use_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Use a tool with the given parameters.
        
        Args:
            tool_name: The name of the tool to use
            parameters: The parameters to pass to the tool
            
        Returns:
            The result of using the tool
            
        Raises:
            NotImplementedError: If the tool is not implemented
        """
        raise NotImplementedError(f"Tool '{tool_name}' is not implemented")
    
    def _extract_tool_use(self, text: str) -> Dict[str, Any]:
        """Extract tool use from the model's response.
        
        Args:
            text: The text to extract tool use from
            
        Returns:
            Dictionary with 'tool_name' and 'parameters' if a tool use is detected,
            None otherwise
        """
        # Default implementation - can be overridden by subclasses
        return None
    
    def _is_code_generation_request(self, text: str) -> bool:
        """Check if the input is requesting code generation.
        
        Args:
            text: The input text to check
            
        Returns:
            True if the input is requesting code generation, False otherwise
        """
        code_keywords = [
            "write", "code", "function", "class", "script", "program",
            "implement", "create", "generate", "example", "how to",
            "snippet", "algorithm", "solve", "solution"
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in code_keywords)
    
    def _is_math_question(self, text: str) -> bool:
        """Check if the input is a math-related question.
        
        Args:
            text: The input text to check
            
        Returns:
            True if the input is a math question, False otherwise
        """
        math_keywords = [
            "calculate", "solve", "equation", "formula", "math", "mathematical",
            "number", "sum", "add", "subtract", "multiply", "divide",
            "algebra", "calculus", "geometry", "trigonometry", "statistics",
            "probability", "derivative", "integral", "matrix", "vector"
        ]
        
        # Check for math operators
        math_operators = r"[+\-*/^=<>]|\d+\s*[+\-*/^=]\s*\d+"
        
        text_lower = text.lower()
        has_math_keyword = any(keyword in text_lower for keyword in math_keywords)
        has_math_operator = bool(re.search(math_operators, text))
        
        return has_math_keyword or has_math_operator
