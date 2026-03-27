"""Math Expert implementation for handling mathematical queries."""

import re
import math
import random
import sympy
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple

from .base_expert import BaseExpertImpl
from ..schemas.base import ExpertDomain
from ..models.expert import ExpertConfig

class MathExpert(BaseExpertImpl):
    """Expert in mathematical problem solving and analysis."""
    
    DOMAIN = ExpertDomain.MATH
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Math Expert.
        
        Args:
            config: Configuration for the expert
        """
        if config is None:
            config = {}
            
        # Set default config values
        config.setdefault("name", "Math Expert")
        config.setdefault("description", (
            "Expert in solving mathematical problems, including algebra, "
            "calculus, statistics, and more. Can explain concepts and "
            "provide step-by-step solutions."
        ))
        config.setdefault("model_name", "gpt-4")
        config.setdefault("temperature", 0.2)
        config.setdefault("max_tokens", 1000)
        
        super().__init__(config)
        
        # Initialize math-related tools and symbols
        self.symbols = {
            'x': sympy.Symbol('x'),
            'y': sympy.Symbol('y'),
            'z': sympy.Symbol('z'),
            't': sympy.Symbol('t'),
            'n': sympy.Symbol('n')
        }
        
        # Supported operations and their handlers
        self.operation_handlers = {
            'simplify': self._simplify_expression,
            'solve': self._solve_equation,
            'factor': self._factor_expression,
            'expand': self._expand_expression,
            'differentiate': self._differentiate,
            'integrate': self._integrate,
            'limit': self._calculate_limit,
            'series': self._calculate_series,
            'evaluate': self._evaluate_expression,
            'graph': self._graph_function,
            'matrix': self._matrix_operations,
            'statistics': self._calculate_statistics,
            'probability': self._calculate_probability,
        }
    
    async def _generate_impl(
        self,
        input_text: str,
        context: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Generate a response to a mathematical query.
        
        Args:
            input_text: The input text containing the mathematical query
            context: Additional context for the expert
            **kwargs: Additional parameters for generation
            
        Returns:
            Dictionary containing the response and metadata
        """
        try:
            # Extract any specific operation from the input
            operation, expression = self._parse_math_query(input_text)
            
            # If we can handle this operation directly, do so
            if operation in self.operation_handlers:
                result = await self.operation_handlers[operation](expression, context)
                return self._format_math_response(result, operation, input_text)
            
            # Otherwise, generate a general response using the language model
            return await self._generate_math_explanation(input_text, context, **kwargs)
            
        except Exception as e:
            self.logger.error(f"Error in MathExpert: {str(e)}", exc_info=True)
            return self._format_error_response(str(e), input_text)
    
    def _parse_math_query(self, text: str) -> Tuple[str, str]:
        """Parse a mathematical query to extract operation and expression.
        
        Args:
            text: The input text to parse
            
        Returns:
            Tuple of (operation, expression)
        """
        text = text.lower().strip()
        
        # Check for specific operation keywords
        for op in self.operation_handlers:
            if text.startswith(f"{op} "):
                return op, text[len(op):].strip()
        
        # Check for common math operation phrases
        op_mapping = {
            'simplify': ['simplify', 'simplification'],
            'solve': ['solve', 'solution', 'solutions', 'find x', 'find y', 'find z'],
            'factor': ['factor', 'factorize', 'factorization'],
            'expand': ['expand', 'expansion'],
            'differentiate': ['derivative', 'differentiate', 'differentiation'],
            'integrate': ['integral', 'integrate', 'integration'],
            'limit': ['limit', 'lim '],
            'series': ['series', 'taylor', 'maclaurin'],
            'evaluate': ['evaluate', 'calculate', 'compute', 'what is', 'what are', 'how much is'],
            'graph': ['graph', 'plot'],
            'matrix': ['matrix', 'matrices', 'determinant', 'inverse', 'eigenvalue'],
            'statistics': ['mean', 'median', 'mode', 'standard deviation', 'variance'],
            'probability': ['probability', 'chance', 'likelihood', 'odds'],
        }
        
        for op, keywords in op_mapping.items():
            if any(keyword in text for keyword in keywords):
                return op, text
        
        # Default to evaluate if no specific operation is found
        return 'evaluate', text
    
    async def _generate_math_explanation(
        self,
        question: str,
        context: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Generate a detailed mathematical explanation or solution.
        
        Args:
            question: The question to answer
            context: Additional context
            **kwargs: Additional parameters
            
        Returns:
            Formatted response with explanation
        """
        # In a real implementation, this would use a language model to generate
        # a detailed explanation. For now, we'll return a simple response.
        
        try:
            # Try to evaluate the expression directly
            result = self._evaluate_expression(question, context)
            explanation = f"The result of '{question}' is {result['result']}."
            
            if 'steps' in result:
                explanation += "\n\nSteps:\n" + "\n".join(f"{i+1}. {step}" for i, step in enumerate(result['steps']))
                
            return self._format_math_response({
                'result': result['result'],
                'explanation': explanation,
                'steps': result.get('steps', [])
            }, 'evaluate', question)
            
        except Exception as e:
            # If direct evaluation fails, provide a general response
            explanation = (
                f"I can help with various mathematical topics including algebra, "
                f"calculus, statistics, and more. For your question about '{question}', "
                f"I would typically provide a detailed solution with steps. "
                f"Please try rephrasing your question or specify the type of "
                f"mathematical operation you'd like to perform."
            )
            
            return self._format_math_response({
                'explanation': explanation,
                'suggested_operations': list(self.operation_handlers.keys())
            }, 'explain', question)
    
    def _evaluate_expression(
        self,
        expression: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate a mathematical expression.
        
        Args:
            expression: The expression to evaluate
            context: Additional context
            
        Returns:
            Dictionary with result and steps
        """
        try:
            # Clean the expression
            expr = expression.strip()
            
            # Handle common replacements
            expr = expr.replace('^', '**')
            
            # Try to evaluate with sympy for symbolic math
            try:
                sympy_expr = sympy.sympify(expr, evaluate=False)
                result = sympy.N(sympy_expr.evalf())
                
                return {
                    'result': str(result),
                    'steps': [f"Evaluated expression: {expr}", f"Result: {result}"],
                    'type': 'exact' if sympy_expr.is_number else 'approximate'
                }
            except Exception:
                pass
            
            # Fall back to eval for simple arithmetic
            allowed_names = {
                **{k: v for k, v in math.__dict__.items() if not k.startswith('_')},
                'e': math.e,
                'pi': math.pi,
                'tau': math.tau,
                'inf': math.inf,
                'j': 1j
            }
            
            # Check for potentially dangerous operations
            if any(op in expr for op in ['import', '__', 'open', 'exec', 'eval']):
                raise ValueError("Invalid operation in expression")
                
            # Evaluate the expression
            result = eval(expr, {"__builtins__": {}}, allowed_names)
            
            return {
                'result': str(result),
                'steps': [f"Evaluated expression: {expr}", f"Result: {result}"],
                'type': 'numeric'
            }
            
        except Exception as e:
            raise ValueError(f"Could not evaluate expression '{expression}': {str(e)}")
    
    def _solve_equation(
        self,
        equation: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Solve an equation for a variable.
        
        Args:
            equation: The equation to solve
            context: Additional context
            
        Returns:
            Dictionary with solution and steps
        """
        try:
            # Parse the equation
            if '=' in equation:
                lhs, rhs = equation.split('=', 1)
                expr = f"{lhs.strip()} - ({rhs.strip()})"
            else:
                expr = equation
                
            # Try to determine the variable to solve for
            variables = set(re.findall(r'\b[a-zA-Z]\b', expr))
            variables.discard('e')  # Don't treat 'e' as a variable
            
            if not variables:
                raise ValueError("No variable to solve for")
                
            # Use the first variable found
            var = sympy.Symbol(next(iter(variables)))
            
            # Solve the equation
            solutions = sympy.solve(expr, var, dict=True)
            
            if not solutions:
                return {
                    'result': "No solution found",
                    'steps': [f"Attempted to solve: {equation}", "No solution found"],
                    'solutions': []
                }
                
            # Format solutions
            formatted_solutions = []
            for sol in solutions:
                if var in sol:
                    formatted_solutions.append(str(sol[var]))
                else:
                    formatted_solutions.append(str(sol))
            
            return {
                'result': f"Solutions: {', '.join(formatted_solutions)}",
                'steps': [
                    f"Solving equation: {equation}",
                    f"Solutions for {var}: {', '.join(formatted_solutions)}"
                ],
                'solutions': formatted_solutions,
                'variable': str(var)
            }
            
        except Exception as e:
            raise ValueError(f"Could not solve equation '{equation}': {str(e)}")
    
    def _differentiate(
        self,
        expression: str,
        context: Dict[str, Any],
        variable: Optional[str] = None,
        n: int = 1
    ) -> Dict[str, Any]:
        """Differentiate an expression.
        
        Args:
            expression: The expression to differentiate
            context: Additional context
            variable: The variable to differentiate with respect to
            n: The order of differentiation
            
        Returns:
            Dictionary with derivative and steps
        """
        try:
            # Parse the expression
            expr = sympy.sympify(expression)
            
            # Determine the variable
            if variable is None:
                variables = expr.free_symbols
                if not variables:
                    raise ValueError("No variable found in expression")
                var = next(iter(variables))
            else:
                var = sympy.Symbol(variable)
            
            # Differentiate
            derivative = sympy.diff(expr, var, n)
            
            return {
                'result': str(derivative),
                'steps': [
                    f"Original: {expr}",
                    f"Derivative with respect to {var} (order {n}): {derivative}"
                ],
                'derivative': str(derivative),
                'variable': str(var),
                'order': n
            }
            
        except Exception as e:
            raise ValueError(f"Could not differentiate expression '{expression}': {str(e)}")
    
    def _integrate(
        self,
        expression: str,
        context: Dict[str, Any],
        variable: Optional[str] = None,
        lower_limit: Optional[Union[str, float]] = None,
        upper_limit: Optional[Union[str, float]] = None
    ) -> Dict[str, Any]:
        """Integrate an expression.
        
        Args:
            expression: The expression to integrate
            context: Additional context
            variable: The variable to integrate with respect to
            lower_limit: The lower limit of integration
            upper_limit: The upper limit of integration
            
        Returns:
            Dictionary with integral and steps
        """
        try:
            # Parse the expression
            expr = sympy.sympify(expression)
            
            # Determine the variable
            if variable is None:
                variables = expr.free_symbols
                if not variables:
                    raise ValueError("No variable found in expression")
                var = next(iter(variables))
            else:
                var = sympy.Symbol(variable)
            
            # Parse limits
            if lower_limit is not None and not isinstance(lower_limit, (int, float)):
                lower_limit = sympy.sympify(lower_limit)
            if upper_limit is not None and not isinstance(upper_limit, (int, float)):
                upper_limit = sympy.sympify(upper_limit)
            
            # Integrate
            if lower_limit is not None and upper_limit is not None:
                # Definite integral
                integral = sympy.integrate(expr, (var, lower_limit, upper_limit))
                result_type = "definite"
                limits = f"from {lower_limit} to {upper_limit}"
            else:
                # Indefinite integral
                integral = sympy.integrate(expr, var)
                result_type = "indefinite"
                limits = ""
            
            return {
                'result': str(integral),
                'steps': [
                    f"Original: {expr}",
                    f"Integral with respect to {var} {limits}: {integral}"
                ],
                'integral': str(integral),
                'variable': str(var),
                'type': result_type,
                'lower_limit': str(lower_limit) if lower_limit is not None else None,
                'upper_limit': str(upper_limit) if upper_limit is not None else None
            }
            
        except Exception as e:
            raise ValueError(f"Could not integrate expression '{expression}': {str(e)}")
    
    def _simplify_expression(
        self,
        expression: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Simplify a mathematical expression.
        
        Args:
            expression: The expression to simplify
            context: Additional context
            
        Returns:
            Dictionary with simplified expression and steps
        """
        try:
            # Parse the expression
            expr = sympy.sympify(expression)
            
            # Simplify
            simplified = sympy.simplify(expr)
            
            return {
                'result': str(simplified),
                'steps': [
                    f"Original: {expr}",
                    f"Simplified: {simplified}"
                ],
                'simplified': str(simplified)
            }
            
        except Exception as e:
            raise ValueError(f"Could not simplify expression '{expression}': {str(e)}")
    
    def _factor_expression(
        self,
        expression: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Factor a mathematical expression.
        
        Args:
            expression: The expression to factor
            context: Additional context
            
        Returns:
            Dictionary with factored expression and steps
        """
        try:
            # Parse the expression
            expr = sympy.sympify(expression)
            
            # Factor
            factored = sympy.factor(expr)
            
            return {
                'result': str(factored),
                'steps': [
                    f"Original: {expr}",
                    f"Factored: {factored}"
                ],
                'factored': str(factored)
            }
            
        except Exception as e:
            raise ValueError(f"Could not factor expression '{expression}': {str(e)}")
    
    def _expand_expression(
        self,
        expression: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Expand a mathematical expression.
        
        Args:
            expression: The expression to expand
            context: Additional context
            
        Returns:
            Dictionary with expanded expression and steps
        """
        try:
            # Parse the expression
            expr = sympy.sympify(expression)
            
            # Expand
            expanded = sympy.expand(expr)
            
            return {
                'result': str(expanded),
                'steps': [
                    f"Original: {expr}",
                    f"Expanded: {expanded}"
                ],
                'expanded': str(expanded)
            }
            
        except Exception as e:
            raise ValueError(f"Could not expand expression '{expression}': {str(e)}")
    
    def _calculate_limit(
        self,
        expression: str,
        context: Dict[str, Any],
        variable: Optional[str] = None,
        point: Optional[Union[str, float]] = None,
        direction: Optional[str] = None
    ) -> Dict[str, Any]:
        """Calculate the limit of an expression.
        
        Args:
            expression: The expression to find the limit of
            context: Additional context
            variable: The variable approaching the limit
            point: The point the variable is approaching
            direction: The direction of the limit ('+' or '-')
            
        Returns:
            Dictionary with limit and steps
        """
        try:
            # Parse the expression
            expr = sympy.sympify(expression.split("as")[0].strip() if "as" in expression else expression)
            
            # Parse the variable and point
            if variable is None or point is None:
                # Try to extract from expression if in the form "limit of ... as x -> a"
                match = re.search(r'limit\s+of\s+(.+?)\s+as\s+([a-zA-Z])\s*[→->]\s*([^\s+-]+)', expression)
                if match:
                    expr = sympy.sympify(match.group(1).strip())
                    variable = match.group(2).strip()
                    point = match.group(3).strip()
                else:
                    # Default to x -> 0
                    variable = 'x'
                    point = '0'
            
            var = sympy.Symbol(variable)
            
            # Parse the point
            if isinstance(point, str):
                if point.lower() == 'inf':
                    point = sympy.oo
                elif point.lower() == '-inf':
                    point = -sympy.oo
                else:
                    point = sympy.sympify(point)
            
            # Calculate the limit
            if direction == '+':
                limit = sympy.limit(expr, var, point, '+')  # type: ignore
            elif direction == '-':
                limit = sympy.limit(expr, var, point, '-')  # type: ignore
            else:
                limit = sympy.limit(expr, var, point)  # type: ignore
            
            return {
                'result': str(limit),
                'steps': [
                    f"Expression: {expr}",
                    f"Limit as {variable} -> {point}: {limit}"
                ],
                'limit': str(limit),
                'variable': str(var),
                'point': str(point),
                'direction': direction
            }
            
        except Exception as e:
            raise ValueError(f"Could not calculate limit for '{expression}': {str(e)}")
    
    def _calculate_series(
        self,
        expression: str,
        context: Dict[str, Any],
        variable: Optional[str] = None,
        point: Optional[Union[str, float]] = None,
        n: int = 5
    ) -> Dict[str, Any]:
        """Calculate the series expansion of an expression.
        
        Args:
            expression: The expression to expand as a series
            context: Additional context
            variable: The variable to expand with respect to
            point: The point around which to expand
            n: The number of terms in the expansion
            
        Returns:
            Dictionary with series expansion and steps
        """
        try:
            # Parse the expression
            expr = sympy.sympify(expression)
            
            # Determine the variable
            if variable is None:
                variables = expr.free_symbols
                if not variables:
                    raise ValueError("No variable found in expression")
                var = next(iter(variables))
            else:
                var = sympy.Symbol(variable)
            
            # Default to Taylor series around 0 (Maclaurin series)
            if point is None:
                point = 0
            elif isinstance(point, str):
                point = sympy.sympify(point)
            
            # Calculate the series expansion
            series = sympy.series(expr, var, point, n=n+1).removeO()
            
            return {
                'result': str(series),
                'steps': [
                    f"Original: {expr}",
                    f"Series expansion around {var} = {point} (first {n} terms): {series}"
                ],
                'series': str(series),
                'variable': str(var),
                'point': str(point),
                'terms': n
            }
            
        except Exception as e:
            raise ValueError(f"Could not calculate series for '{expression}': {str(e)}")
    
    def _graph_function(
        self,
        expression: str,
        context: Dict[str, Any],
        variable: Optional[str] = None,
        x_range: Tuple[float, float] = (-10, 10),
        y_range: Optional[Tuple[float, float]] = None
    ) -> Dict[str, Any]:
        """Generate a graph of a function.
        
        Args:
            expression: The function to graph
            context: Additional context
            variable: The independent variable
            x_range: The range of x values to plot
            y_range: Optional range of y values to plot
            
        Returns:
            Dictionary with graph data and steps
        """
        try:
            # In a real implementation, this would generate an image or return plot data
            # For now, we'll return a description of what would be plotted
            
            # Parse the expression
            expr = sympy.sympify(expression)
            
            # Determine the variable
            if variable is None:
                variables = expr.free_symbols
                if not variables:
                    raise ValueError("No variable found in expression")
                var = next(iter(variables))
            else:
                var = sympy.Symbol(variable)
            
            return {
                'result': f"Graph of {expr} with respect to {var} from {x_range[0]} to {x_range[1]}",
                'steps': [
                    f"Function: {expr}",
                    f"Variable: {var}",
                    f"X-range: {x_range}",
                    "Graph would be displayed here in an interactive environment"
                ],
                'function': str(expr),
                'variable': str(var),
                'x_range': x_range,
                'y_range': y_range,
                'type': 'graph'
            }
            
        except Exception as e:
            raise ValueError(f"Could not generate graph for '{expression}': {str(e)}")
    
    def _matrix_operations(
        self,
        operation: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform matrix operations.
        
        Args:
            operation: The matrix operation to perform
            context: Additional context
            
        Returns:
            Dictionary with operation result and steps
        """
        try:
            # This is a simplified implementation
            # In a real implementation, you would parse the operation and perform it
            
            return {
                'result': f"Result of matrix operation: {operation}",
                'steps': [
                    f"Performed matrix operation: {operation}",
                    "Matrix operations would be performed here"
                ],
                'operation': operation,
                'type': 'matrix_operation'
            }
            
        except Exception as e:
            raise ValueError(f"Could not perform matrix operation '{operation}': {str(e)}")
    
    def _calculate_statistics(
        self,
        data: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate statistics for a dataset.
        
        Args:
            data: The data to analyze (can be a list or description)
            context: Additional context
            
        Returns:
            Dictionary with statistics and steps
        """
        try:
            # Try to parse the data as a list of numbers
            try:
                # Handle different input formats
                if '[' in data and ']' in data:
                    # Extract list from string like "[1, 2, 3, 4, 5]"
                    data_str = data[data.find('[')+1:data.rfind(']')]
                    numbers = [float(x.strip()) for x in data_str.split(',') if x.strip()]
                else:
                    # Handle space or comma separated values
                    numbers = [float(x) for x in re.findall(r'[\d.]+', data)]
                
                if not numbers:
                    raise ValueError("No numeric data found")
                
                # Calculate statistics
                import statistics
                
                stats = {
                    'count': len(numbers),
                    'mean': statistics.mean(numbers),
                    'median': statistics.median(numbers),
                    'mode': statistics.mode(numbers) if len(numbers) == len(set(numbers)) else "No unique mode",
                    'stdev': statistics.stdev(numbers) if len(numbers) > 1 else 0,
                    'variance': statistics.variance(numbers) if len(numbers) > 1 else 0,
                    'min': min(numbers),
                    'max': max(numbers),
                    'sum': sum(numbers)
                }
                
                return {
                    'result': f"Statistics for dataset with {len(numbers)} values: {stats}",
                    'steps': [
                        f"Dataset: {numbers}",
                        f"Calculated statistics: {stats}"
                    ],
                    'statistics': stats,
                    'data': numbers
                }
                
            except (ValueError, IndexError):
                # If we can't parse as numbers, return a general response
                return {
                    'result': f"Could not parse numeric data from '{data}'. Please provide a list of numbers.",
                    'steps': [
                        f"Could not parse numeric data from: {data}",
                        "Please provide data in a format like: 1, 2, 3, 4, 5 or [1, 2, 3, 4, 5]"
                    ]
                }
                
        except Exception as e:
            raise ValueError(f"Could not calculate statistics for '{data}': {str(e)}")
    
    def _calculate_probability(
        self,
        problem: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Solve a probability problem.
        
        Args:
            problem: The probability problem to solve
            context: Additional context
            
        Returns:
            Dictionary with solution and steps
        """
        try:
            # This is a simplified implementation
            # In a real implementation, you would parse the problem and solve it
            
            return {
                'result': f"Solution to probability problem: {problem}",
                'steps': [
                    f"Probability problem: {problem}",
                    "Probability calculation would be performed here"
                ],
                'problem': problem,
                'type': 'probability'
            }
            
        except Exception as e:
            raise ValueError(f"Could not solve probability problem '{problem}': {str(e)}")
    
    def _format_math_response(
        self,
        result: Dict[str, Any],
        operation: str,
        original_input: str
    ) -> Dict[str, Any]:
        """Format a mathematical response with metadata.
        
        Args:
            result: The result of the mathematical operation
            operation: The operation that was performed
            original_input: The original input from the user
            
        Returns:
            Formatted response dictionary
        """
        # Extract the main result
        response_text = result.get('result', 'No result')
        
        # Add explanation if available
        if 'explanation' in result:
            response_text = f"{result['explanation']}\n\nResult: {response_text}"
        
        # Add steps if available
        if 'steps' in result and result['steps']:
            steps = "\n".join(f"- {step}" for step in result['steps'])
            response_text = f"{response_text}\n\nSteps:\n{steps}"
        
        # Format the final response
        return {
            'response': response_text,
            'metadata': {
                'operation': operation,
                'original_input': original_input,
                'expert': self.config.name,
                'domain': self.DOMAIN.value,
                'timestamp': datetime.now().isoformat(),
                **{k: v for k, v in result.items() if k not in ('result', 'explanation', 'steps')}
            },
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
                f"I encountered an error while processing your request:\n"
                f"{error_message}\n\n"
                f"Please try rephrasing your question or ask about a different mathematical topic."
            ),
            'metadata': {
                'error': error_message,
                'original_input': original_input,
                'expert': self.config.name,
                'domain': self.DOMAIN.value,
                'timestamp': datetime.now().isoformat()
            },
            'sources': []
        }
