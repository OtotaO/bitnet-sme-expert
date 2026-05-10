"""Code Expert implementation for handling programming and software development queries."""

import re
import ast
import json
import inspect
import textwrap
import subprocess
import sys
import os
from typing import Dict, Any, List, Optional, Tuple, Union, Type, Callable
from pathlib import Path
from datetime import datetime

from .base_expert import BaseExpertImpl
from ..schemas.base import ExpertDomain
from ..models.expert import ExpertConfig

class CodeExpert(BaseExpertImpl):
    """Expert in programming, code generation, and software development."""
    
    DOMAIN = ExpertDomain.CODE
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Code Expert.
        
        Args:
            config: Configuration for the expert
        """
        if config is None:
            config = {}
            
        # Set default config values
        config.setdefault("name", "Code Expert")
        config.setdefault("description", (
            "Expert in programming, code generation, debugging, and software development. "
            "Can help with multiple programming languages, algorithms, data structures, "
            "and software design patterns."
        ))
        config.setdefault("model_name", "gpt-4")
        config.setdefault("temperature", 0.3)
        config.setdefault("max_tokens", 2000)
        
        # Supported programming languages with their file extensions and comment syntax
        self.supported_languages = {
            'python': {
                'extensions': ['.py'],
                'comment': '#',
                'docstring': '\"\"\"{content}\"\"\"',
            },
            'javascript': {
                'extensions': ['.js', '.jsx', '.ts', '.tsx'],
                'comment': '//',
                'docstring': '/**\n * {content}\n */',
            },
            'java': {
                'extensions': ['.java'],
                'comment': '//',
                'docstring': '/**\n * {content}\n */',
            },
            'c': {
                'extensions': ['.c', '.h'],
                'comment': '//',
                'docstring': '/*\n * {content}\n */',
            },
            'cpp': {
                'extensions': ['.cpp', '.hpp', '.cc', '.hxx'],
                'comment': '//',
                'docstring': '/*\n * {content}\n */',
            },
            'csharp': {
                'extensions': ['.cs'],
                'comment': '//',
                'docstring': '/// <summary>\n/// {content}\n/// </summary>',
            },
            'go': {
                'extensions': ['.go'],
                'comment': '//',
                'docstring': '/*\n{content}\n*/',
            },
            'rust': {
                'extensions': ['.rs'],
                'comment': '//',
                'docstring': '/// {content}',
            },
            'ruby': {
                'extensions': ['.rb'],
                'comment': '#',
                'docstring': '=begin\n{content}\n=end',
            },
            'php': {
                'extensions': ['.php'],
                'comment': '//',
                'docstring': '/**\n * {content}\n */',
            },
            'swift': {
                'extensions': ['.swift'],
                'comment': '//',
                'docstring': '/// {content}',
            },
            'kotlin': {
                'extensions': ['.kt', '.kts'],
                'comment': '//',
                'docstring': '/**\n * {content}\n */',
            },
        }
        
        # Common code patterns and their handlers
        self.code_handlers = {
            'generate': self._handle_code_generation,
            'explain': self._explain_code,
            'debug': self._debug_code,
            'optimize': self._optimize_code,
            'refactor': self._refactor_code,
            'test': self._write_tests,
            'convert': self._convert_code,
            'document': self._document_code,
        }
        
        super().__init__(config)
    
    async def _generate_impl(
        self,
        input_text: str,
        context: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Generate a response to a programming-related query.
        
        Args:
            input_text: The input text containing the programming query
            context: Additional context for the expert
            **kwargs: Additional parameters for generation
            
        Returns:
            Dictionary containing the response and metadata
        """
        try:
            # Determine the type of code-related request
            request_type, details = self._parse_code_request(input_text)
            
            # Handle the request using the appropriate handler
            if request_type in self.code_handlers:
                result = await self.code_handlers[request_type](details, context, **kwargs)
                return self._format_code_response(result, request_type, input_text)
            else:
                # Default to code generation if no specific handler is found
                result = await self._handle_code_generation(
                    {'code': input_text, 'language': None}, 
                    context, 
                    **kwargs
                )
                return self._format_code_response(result, 'generate', input_text)
                
        except Exception as e:
            self.logger.error(f"Error in CodeExpert: {str(e)}", exc_info=True)
            return self._format_error_response(str(e), input_text)
    
    def _parse_code_request(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """Parse a code-related request to determine the type and extract details.
        
        Args:
            text: The input text to parse
            
        Returns:
            Tuple of (request_type, details_dict)
        """
        text_lower = text.lower().strip()
        
        # Check for specific request types
        for req_type in self.code_handlers:
            if text_lower.startswith(f"{req_type} ") or f" {req_type} " in f" {text_lower} ":
                # Extract the code or details after the request type
                details = text[text.lower().find(req_type) + len(req_type):].strip()
                return req_type, {'code': details, 'language': None}
        
        # Check for language-specific requests (e.g., "python code to sort a list")
        for lang in self.supported_languages:
            if f"{lang} code" in text_lower or f"{lang} function" in text_lower:
                return 'generate', {'code': text, 'language': lang}
        
        # Default to code generation
        return 'generate', {'code': text, 'language': None}
    
    async def _handle_code_generation(
        self,
        details: Dict[str, Any],
        context: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Handle code generation requests.
        
        Args:
            details: Dictionary containing 'code' and 'language' keys
            context: Additional context
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with generated code and metadata
        """
        code_request = details.get('code', '').strip()
        language = details.get('language')
        
        # If no language specified, try to detect from the request
        if not language:
            language = self._detect_language(code_request)
        
        # In a real implementation, this would use a language model to generate code
        # For now, we'll return a simple response
        
        # Generate a simple code example based on the request
        if 'sort' in code_request.lower() and 'list' in code_request.lower():
            if language == 'python':
                code = "# Sort a list in Python\nmy_list = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]\nsorted_list = sorted(my_list)\nprint(f\"Original: {my_list}\")\nprint(f\"Sorted: {sorted_list}\")"
            elif language == 'javascript':
                code = "// Sort an array in JavaScript\nconst myArray = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5];\nconst sortedArray = [...myArray].sort((a, b) => a - b);\nconsole.log(`Original: ${myArray}`);\nconsole.log(`Sorted: ${sortedArray}`);"
            else:
                code = f"// Sorting implementation in {language or 'your language'}\n// Code would be generated here based on the language"
        elif 'fibonacci' in code_request.lower():
            if language == 'python':
                code = "def fibonacci(n):\n    \"\"\"Generate the first n Fibonacci numbers.\"\"\"\n    a, b = 0, 1\n    result = []\n    for _ in range(n):\n        result.append(a)\n        a, b = b, a + b\n    return result\n\n# Example usage\nprint(fibonacci(10))  # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]"
            elif language == 'javascript':
                code = "/**\n * Generate the first n Fibonacci numbers.\n * @param {number} n - The number of Fibonacci numbers to generate\n * @returns {number[]} Array of Fibonacci numbers\n */\nfunction fibonacci(n) {\n    let a = 0, b = 1;\n    const result = [];\n    for (let i = 0; i < n; i++) {\n        result.push(a);\n        [a, b] = [b, a + b];\n    }\n    return result;\n}\n\n// Example usage\nconsole.log(fibonacci(10));  // [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]"
            else:
                code = f"// Fibonacci sequence implementation in {language or 'your language'}\n// Code would be generated here based on the language"
        else:
            # Generic response
            lang_display = language if language else 'a programming language'
            code = f"# Code generation for: {code_request}\n# This is a placeholder for the actual code that would be generated\n# The implementation would be in {lang_display}\n\n# Your code would be generated here"
            
            # Add language-specific boilerplate if known
            if language == 'python':
                code = f"""# {code_request}

def main():
    # Your code here
    pass

if __name__ == "__main__":
    main()"""
            elif language == 'javascript':
                code = f"// {code_request}\n\nfunction main() {{\n    // Your code here\n}}\n\n// Run the main function\nmain();"
        
        return {
            'code': code,
            'language': language or 'text',
            'explanation': f"Generated code for: {code_request}",
            'type': 'code_generation',
            'success': True
        }
    
    async def _explain_code(
        self,
        details: Dict[str, Any],
        context: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Explain what a piece of code does.
        
        Args:
            details: Dictionary containing 'code' and 'language' keys
            context: Additional context
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with code explanation and metadata
        """
        code = details.get('code', '').strip()
        language = details.get('language') or self._detect_language(code)
        
        # In a real implementation, this would analyze the code and generate an explanation
        # For now, we'll return a simple response
        
        explanation = (
            f"This appears to be a code snippet in {language or 'an unknown programming language'}. "
            f"In a real implementation, I would analyze the code and provide a detailed explanation "
            f"of what it does, how it works, and any important concepts it demonstrates.\n\n"
            f"The code is:\n```{language or ''}\n{code}\n```"
        )
        
        return {
            'explanation': explanation,
            'code': code,
            'language': language,
            'type': 'code_explanation',
            'success': True
        }
    
    async def _debug_code(
        self,
        details: Dict[str, Any],
        context: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Debug a piece of code.
        
        Args:
            details: Dictionary containing 'code', 'language', and 'error' keys
            context: Additional context
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with debugging information and fixed code if applicable
        """
        code = details.get('code', '').strip()
        language = details.get('language') or self._detect_language(code)
        error = details.get('error', '')
        
        # In a real implementation, this would analyze the code and error
        # For now, we'll return a simple response
        
        explanation = (
            f"Debugging code in {language or 'an unknown language'}.\n"
        )
        
        if error:
            explanation += f"Error reported: {error}\n\n"
        
        explanation += (
            "In a real implementation, I would analyze the code, identify potential issues, "
            "and suggest fixes. I would look for common mistakes like syntax errors, "
            "logical errors, or incorrect function usage."
        )
        
        return {
            'explanation': explanation,
            'code': code,
            'language': language,
            'error': error,
            'suggested_fixes': ["Check for syntax errors", "Verify variable names", "Review function signatures"],
            'type': 'code_debugging',
            'success': True
        }
    
    async def _optimize_code(
        self,
        details: Dict[str, Any],
        context: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Optimize a piece of code.
        
        Args:
            details: Dictionary containing 'code' and 'language' keys
            context: Additional context
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with optimized code and explanation
        """
        code = details.get('code', '').strip()
        language = details.get('language') or self._detect_language(code)
        
        # In a real implementation, this would analyze the code for optimization opportunities
        # For now, we'll return a simple response
        
        explanation = (
            f"Optimizing code in {language or 'an unknown language'}.\n\n"
            "Potential optimizations might include:\n"
            "- Using more efficient algorithms or data structures\n"
            "- Reducing time complexity (e.g., from O(n²) to O(n log n))\n"
            "- Reducing space complexity\n"
            "- Using built-in functions or libraries\n"
            "- Eliminating redundant calculations"
        )
        
        return {
            'explanation': explanation,
            'original_code': code,
            'optimized_code': f"# Optimized version of the code would be shown here\n# Original code: {code[:100]}...",
            'language': language,
            'optimizations': ["Algorithm optimization", "Memory usage", "Performance improvements"],
            'type': 'code_optimization',
            'success': True
        }
    
    async def _refactor_code(
        self,
        details: Dict[str, Any],
        context: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Refactor a piece of code.
        
        Args:
            details: Dictionary containing 'code' and 'language' keys
            context: Additional context
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with refactored code and explanation
        """
        code = details.get('code', '').strip()
        language = details.get('language') or self._detect_language(code)
        
        # In a real implementation, this would analyze and refactor the code
        # For now, we'll return a simple response
        
        explanation = (
            f"Refactoring code in {language or 'an unknown language'}.\n\n"
            "Refactoring improvements might include:\n"
            "- Improving code readability and maintainability\n"
            "- Extracting repeated code into functions\n"
            "- Renaming variables for clarity\n"
            "- Simplifying complex conditionals\n"
            "- Applying design patterns where appropriate"
        )
        
        return {
            'explanation': explanation,
            'original_code': code,
            'refactored_code': f"# Refactored version of the code would be shown here\n# Original code: {code[:100]}...",
            'language': language,
            'refactoring_techniques': ["Extract Method", "Rename Variable", "Simplify Conditional"],
            'type': 'code_refactoring',
            'success': True
        }
    
    async def _write_tests(
        self,
        details: Dict[str, Any],
        context: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Write tests for a piece of code.
        
        Args:
            details: Dictionary containing 'code' and 'language' keys
            context: Additional context
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with test code and explanation
        """
        code = details.get('code', '').strip()
        language = details.get('language') or self._detect_language(code)
        
        # In a real implementation, this would analyze the code and generate appropriate tests
        # For now, we'll return a simple response
        
        test_code = f"""# Test cases for the provided code
# In a real implementation, this would include actual test cases
# that verify the functionality of the code

def test_example():
    # Test case 1: Basic functionality
    # result = your_function(input)
    # assert result == expected_output
    pass

def test_edge_cases():
    # Test case 2: Edge cases
    # result = your_function(edge_case_input)
    # assert result == expected_output
    pass

# Add more test cases as needed"""
        
        explanation = (
            f"Generated test cases for code in {language or 'an unknown language'}.\n\n"
            "The test suite includes:\n"
            "- Basic functionality tests\n"
            "- Edge case tests\n"
            "- Error handling tests\n\n"
            "In a real implementation, the tests would be tailored to the specific code "
            "and would include assertions to verify the expected behavior."
        )
        
        return {
            'explanation': explanation,
            'original_code': code,
            'test_code': test_code,
            'language': language or 'python',  # Default to Python for test examples
            'test_frameworks': ['pytest', 'unittest', 'doctest'],
            'type': 'test_generation',
            'success': True
        }
    
    async def _convert_code(
        self,
        details: Dict[str, Any],
        context: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Convert code from one language to another.
        
        Args:
            details: Dictionary containing 'code', 'source_language', and 'target_language' keys
            context: Additional context
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with converted code and explanation
        """
        code = details.get('code', '').strip()
        source_lang = details.get('source_language') or self._detect_language(code)
        target_lang = details.get('target_language')
        
        if not target_lang:
            # Default to Python if no target language is specified
            target_lang = 'python' if source_lang != 'python' else 'javascript'
        
        # In a real implementation, this would perform the actual code conversion
        # For now, we'll return a simple response
        
        converted_code = f"// Code converted from {source_lang or 'source language'} to {target_lang}\n// In a real implementation, this would be the actual converted code\n\n// Original code in {source_lang}:\n// {code[:200]}{'...' if len(code) > 200 else ''}"
        
        explanation = (
            f"Converted code from {source_lang or 'source language'} to {target_lang}.\n\n"
            "In a real implementation, this would include the actual converted code "
            "with equivalent functionality in the target language, along with any necessary "
            "adjustments for language-specific features and idioms."
        )
        
        return {
            'explanation': explanation,
            'original_code': code,
            'converted_code': converted_code,
            'source_language': source_lang,
            'target_language': target_lang,
            'type': 'code_conversion',
            'success': True
        }
    
    async def _document_code(
        self,
        details: Dict[str, Any],
        context: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Generate documentation for a piece of code.
        
        Args:
            details: Dictionary containing 'code' and 'language' keys
            context: Additional context
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with documented code and explanation
        """
        code = details.get('code', '').strip()
        language = details.get('language') or self._detect_language(code)
        
        # In a real implementation, this would analyze the code and generate documentation
        # For now, we'll return a simple response
        
        documented_code = f"""
Module/Function Documentation

In a real implementation, this would include:
- Module/function/class documentation
- Parameter descriptions
- Return value descriptions
- Examples
- Any relevant notes or warnings
"""
        
        if language == 'python':
            documented_code += '\n\n' + '\n'.join([
                'def example_function(param1, param2):',
                '    """Example function with documentation.',
                '    ', 
                '    Args:',
                '        param1: Description of param1',
                '        param2: Description of param2',
                '        ', 
                '    Returns:',
                '        Description of return value',
                '        ', 
                '    Example:',
                '        >>> result = example_function(1, 2)',
                '        >>> print(result)',
                '    """',
                '    return param1 + param2'
            ])
        elif language == 'javascript':
            documented_code += '\n\n' + '\n'.join([
                '/**',
                ' * Example function with JSDoc documentation.',
                ' * @param {number} param1 - Description of param1',
                ' * @param {string} param2 - Description of param2',
                ' * @returns {boolean} Description of return value',
                ' * @example',
                ' * // returns true',
                ' * const result = exampleFunction(1, "test");',
                ' */',
                'function exampleFunction(param1, param2) {',
                '    return true;',
                '}'
            ])
        
        explanation = (
            f"Generated documentation for code in {language or 'an unknown language'}.\n\n"
            "The documentation includes:\n"
            "- Function/method signatures\n"
            "- Parameter descriptions\n"
            "- Return value descriptions\n"
            "- Examples\n"
            "- Any relevant notes or warnings"
        )
        
        return {
            'explanation': explanation,
            'original_code': code,
            'documented_code': documented_code,
            'language': language,
            'documentation_style': 'docstrings' if language == 'python' else 'JSDoc' if language == 'javascript' else 'standard',
            'type': 'code_documentation',
            'success': True
        }
    
    def _detect_language(self, code: str) -> Optional[str]:
        """Detect the programming language of a code snippet.
        
        Args:
            code: The code to analyze
            
        Returns:
            Detected language or None if unknown
        """
        if not code.strip():
            return None
            
        # Check for language-specific patterns
        code_lower = code.lower()
        
        # Check for Python shebang or specific Python syntax
        if code_lower.startswith('#!/usr/bin/env python') or 'import ' in code_lower or 'def ' in code_lower:
            return 'python'
            
        # Check for JavaScript/TypeScript
        if 'function ' in code_lower or 'const ' in code_lower or 'let ' in code_lower or 'var ' in code_lower:
            if 'interface ' in code_lower or 'type ' in code_lower or ' as ' in code_lower:
                return 'typescript' if 'ts' in code_lower else 'typescript'
            return 'javascript'
            
        # Check for Java
        if 'public class ' in code_lower or 'public static void main' in code_lower:
            return 'java'
            
        # Check for C/C++
        if '#include ' in code_lower or 'int main(' in code_lower:
            if 'using namespace ' in code_lower or 'std::' in code_lower or 'new ' in code_lower:
                return 'cpp'
            return 'c'
            
        # Check for C#
        if 'using ' in code_lower and ';' in code_lower and ('class ' in code_lower or 'namespace ' in code_lower):
            return 'csharp'
            
        # Check for Go
        if 'package ' in code_lower and 'import (' in code_lower and 'func ' in code_lower:
            return 'go'
            
        # Check for Ruby
        if 'def ' in code_lower and 'end' in code_lower and ('puts ' in code_lower or 'require ' in code_lower):
            return 'ruby'
            
        # Check for PHP
        if '<?php' in code_lower or '$_' in code_lower or '->' in code_lower:
            return 'php'
            
        # Check for Swift
        if 'import ' in code_lower and ('func ' in code_lower or 'var ' in code_lower) and ';' not in code_lower:
            return 'swift'
            
        # Check for Kotlin
        if ('fun ' in code_lower or 'val ' in code_lower or 'var ' in code_lower) and ';' not in code_lower:
            return 'kotlin'
            
        # Default to Python if we can't determine the language
        return 'python'
    
    def _format_code_response(
        self,
        result: Dict[str, Any],
        operation: str,
        original_input: str
    ) -> Dict[str, Any]:
        """Format a code-related response with metadata.
        
        Args:
            result: The result of the code operation
            operation: The operation that was performed
            original_input: The original input from the user
            
        Returns:
            Formatted response dictionary
        """
        # Build the response text
        response_parts = []
        
        # Add explanation if available
        if 'explanation' in result:
            response_parts.append(result['explanation'])
        
        # Add code blocks if available
        if 'code' in result and result['code']:
            lang = result.get('language', 'text')
            response_parts.append(f"```{lang}\n{result['code']}\n```")
            
        if 'optimized_code' in result:
            response_parts.append("\nOptimized code:")
            lang = result.get('language', 'text')
            response_parts.append(f"```{lang}\n{result['optimized_code']}\n```")
            
        if 'refactored_code' in result:
            response_parts.append("\nRefactored code:")
            lang = result.get('language', 'text')
            response_parts.append(f"```{lang}\n{result['refactored_code']}\n```")
            
        if 'test_code' in result:
            response_parts.append("\nTest code:")
            lang = result.get('language', 'python')
            response_parts.append(f"```{lang}\n{result['test_code']}\n```")
            
        if 'converted_code' in result:
            response_parts.append(
                f"\nConverted code ({result.get('source_language', 'source')} -> {result.get('target_language', 'target')}):"
            )
            lang = result.get('target_language', 'text')
            response_parts.append(f"```{lang}\n{result['converted_code']}\n```")
            
        if 'documented_code' in result:
            response_parts.append("\nDocumented code:")
            lang = result.get('language', 'text')
            response_parts.append(f"```{lang}\n{result['documented_code']}\n```")
        
        # Join all parts with double newlines
        response_text = '\n\n'.join(str(part) for part in response_parts if part)
        
        # Prepare metadata
        metadata = {
            'operation': operation,
            'original_input': original_input,
            'expert': self.config.name,
            'domain': self.DOMAIN.value,
            'timestamp': datetime.now().isoformat(),
        }
        
        # Add any additional metadata from the result
        for key in ['language', 'type', 'success', 'optimizations', 'refactoring_techniques', 
                   'test_frameworks', 'source_language', 'target_language', 'documentation_style']:
            if key in result:
                metadata[key] = result[key]
        
        return {
            'response': response_text,
            'metadata': metadata,
            'sources': result.get('sources', [])
        }
    
    def _format_error_response(
        self,
        error_message: str,
        original_input: str
    ) -> Dict[str, Any]:
        """Format an error response.
        
        Args:
            error_message: The error message
            original_input: The original input from the user
            
        Returns:
            Formatted error response
        """
        return {
            'response': (
                f"I encountered an error while processing your code request:\n"
                f"{error_message}\n\n"
                f"Please check your input and try again. If the problem persists, "
                f"you may need to provide more context or rephrase your request."
            ),
            'metadata': {
                'error': error_message,
                'original_input': original_input,
                'expert': self.config.name,
                'domain': self.DOMAIN.value,
                'timestamp': datetime.now().isoformat(),
                'success': False
            },
            'sources': []
        }
