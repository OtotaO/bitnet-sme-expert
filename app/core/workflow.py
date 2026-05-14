import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from enum import StrEnum
from functools import wraps
from typing import Any, TypeVar
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

T = TypeVar("T")


class WorkflowStatus(StrEnum):
    """Status of a workflow."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowStepStatus(StrEnum):
    """Status of a workflow step."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStepResult(BaseModel):
    """Result of a workflow step execution."""

    step_name: str
    status: WorkflowStepStatus = WorkflowStepStatus.PENDING
    start_time: datetime | None = None
    end_time: datetime | None = None
    duration: float | None = None
    error: str | None = None
    result: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkflowContext(BaseModel):
    """Context for workflow execution."""

    workflow_id: str = Field(default_factory=lambda: f"wf_{uuid4().hex[:8]}")
    status: WorkflowStatus = WorkflowStatus.PENDING
    start_time: datetime | None = None
    end_time: datetime | None = None
    current_step: str | None = None
    steps_completed: list[str] = Field(default_factory=list)
    steps_failed: list[str] = Field(default_factory=list)
    steps_skipped: list[str] = Field(default_factory=list)
    results: dict[str, WorkflowStepResult] = Field(default_factory=dict)
    data: dict[str, Any] = Field(default_factory=dict)
    errors: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkflowError(Exception):
    """Base exception for workflow-related errors."""

    pass


class WorkflowStepError(WorkflowError):
    """Exception raised when a workflow step fails."""

    def __init__(self, step_name: str, message: str, original_error: Exception | None = None):
        self.step_name = step_name
        self.original_error = original_error
        super().__init__(f"Step '{step_name}' failed: {message}")


class StepDependencyError(WorkflowError):
    """Exception raised when step dependencies are not met."""

    pass


class WorkflowStep:
    """Represents a step in a workflow."""

    def __init__(
        self,
        func: Callable[..., Awaitable[dict[str, Any]]],
        name: str | None = None,
        description: str | None = None,
        input_schema: Any | None = None,
        output_schema: Any | None = None,
        timeout: float | None = None,
        retries: int = 0,
        retry_delay: float = 1.0,
        requires: list[str] | None = None,
        provides: list[str] | None = None,
        enabled: bool = True,
    ):
        """Initialize a workflow step.

        Args:
            func: The async function to execute
            name: Name of the step (defaults to function name)
            description: Description of what the step does
            input_schema: Expected input schema (Pydantic model)
            output_schema: Expected output schema (Pydantic model)
            timeout: Maximum execution time in seconds
            retries: Number of retry attempts on failure
            retry_delay: Delay between retries in seconds
            requires: List of step names that must complete before this step
            provides: List of output keys this step provides
            enabled: Whether the step is enabled
        """
        self.func = func
        self.name = name or func.__name__
        self.description = description or func.__doc__ or ""
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.timeout = timeout
        self.retries = max(0, retries)
        self.retry_delay = max(0.0, retry_delay)
        self.requires = requires or []
        self.provides = provides or []
        self.enabled = enabled
        self.logger = logger.getChild(f"step.{self.name}")

    async def execute(self, context: WorkflowContext) -> dict[str, Any]:
        """Execute the step with the given context."""
        if not self.enabled:
            self.logger.debug(f"Step '{self.name}' is disabled, skipping")
            return {}

        result = WorkflowStepResult(
            step_name=self.name,
            status=WorkflowStepStatus.RUNNING,
            start_time=datetime.now(UTC),
            metadata={"retries": 0, "attempts": 0, "timeout": self.timeout},
        )

        context.results[self.name] = result
        context.current_step = self.name

        self.logger.info(f"Executing step: {self.name}")

        last_error = None

        for attempt in range(1, self.retries + 2):  # +1 for the initial attempt
            result.metadata["attempts"] = attempt

            if attempt > 1:
                self.logger.warning(
                    f"Retry {attempt - 1}/{self.retries} for step '{self.name}'",
                    extra={"error": str(last_error) if last_error else "Unknown error"},
                )
                await asyncio.sleep(self.retry_delay * (2 ** (attempt - 2)))  # Exponential backoff

            try:
                # Validate input if schema is provided
                if self.input_schema:
                    try:
                        self.input_schema(**context.data)
                    except Exception as e:
                        raise ValueError(f"Invalid input: {e!s}") from e

                # Execute the step function
                if self.timeout:
                    step_result = await asyncio.wait_for(self.func(context), timeout=self.timeout)
                else:
                    step_result = await self.func(context)

                # Validate output if schema is provided
                if self.output_schema and step_result is not None:
                    try:
                        if isinstance(step_result, dict):
                            step_result = self.output_schema(**step_result)
                        else:
                            step_result = self.output_schema(step_result)
                    except Exception as e:
                        raise ValueError(f"Invalid output: {e!s}") from e

                # Update step result
                result.status = WorkflowStepStatus.COMPLETED
                result.result = (
                    step_result if isinstance(step_result, dict) else {"result": step_result}
                )

                # Update context with step results
                if step_result and isinstance(step_result, dict):
                    context.data.update(step_result)

                self.logger.info(f"Step '{self.name}' completed successfully")
                return step_result or {}

            except TimeoutError:
                error_msg = f"Step '{self.name}' timed out after {self.timeout}s"
                last_error = TimeoutError(error_msg)
                result.metadata["timeout"] = True

            except Exception as e:
                last_error = e
                if attempt <= self.retries:
                    result.metadata["retries"] += 1
                self.logger.warning(
                    f"Attempt {attempt} failed for step '{self.name}'", exc_info=True
                )

        # If we get here, all attempts failed
        error_msg = f"Step '{self.name}' failed after {self.retries + 1} attempts"
        if last_error:
            error_msg += f": {last_error!s}"

        result.status = WorkflowStepStatus.FAILED
        result.error = error_msg
        result.metadata["error"] = str(last_error) if last_error else "Unknown error"

        self.logger.error(error_msg, exc_info=last_error)
        raise WorkflowStepError(self.name, error_msg, last_error)

    def __call__(self, context: WorkflowContext) -> Awaitable[dict[str, Any]]:
        """Allow the step to be called directly."""
        return self.execute(context)


def step(
    name: str | None = None,
    description: str | None = None,
    input_schema: Any | None = None,
    output_schema: Any | None = None,
    timeout: float | None = None,
    retries: int = 0,
    retry_delay: float = 1.0,
    requires: list[str] | None = None,
    provides: list[str] | None = None,
    enabled: bool = True,
):
    """Decorator for defining workflow steps with metadata.

    Args:
        name: Name of the step (defaults to function name)
        description: Description of what the step does
        input_schema: Expected input schema (Pydantic model)
        output_schema: Expected output schema (Pydantic model)
        timeout: Maximum execution time in seconds
        retries: Number of retry attempts on failure
        retry_delay: Delay between retries in seconds
        requires: List of step names that must complete before this step
        provides: List of output keys this step provides
        enabled: Whether the step is enabled
    """

    def decorator(func):
        step_metadata = {
            "name": name or func.__name__,
            "description": description or func.__doc__ or "",
            "input_schema": input_schema,
            "output_schema": output_schema,
            "timeout": timeout,
            "retries": retries,
            "retry_delay": retry_delay,
            "requires": requires or [],
            "provides": provides or [],
            "enabled": enabled,
        }

        @wraps(func)
        async def wrapper(context: WorkflowContext, *args, **kwargs):
            # If called directly, create a minimal context if not provided
            if not isinstance(context, WorkflowContext):
                context = WorkflowContext()
            return await func(context, *args, **kwargs)

        wrapper._step_metadata = step_metadata
        return wrapper

    return decorator


class Workflow:
    """Orchestrates the execution of workflow steps."""

    def __init__(
        self,
        name: str,
        description: str | None = None,
        max_concurrent: int | None = None,
        on_error: str = "stop",  # 'stop', 'continue', 'cancel'
    ):
        """Initialize a workflow.

        Args:
            name: Name of the workflow
            description: Description of the workflow
            max_concurrent: Maximum number of concurrent steps (None for no limit)
            on_error: Behavior on error ('stop', 'continue', 'cancel')
        """
        self.name = name
        self.description = description or ""
        self.max_concurrent = max_concurrent
        self.on_error = on_error
        self.steps: list[WorkflowStep] = []
        self.logger = logger.getChild(f"workflow.{name}")

    def add_step(
        self,
        func: Callable[..., Awaitable[dict[str, Any]]],
        name: str | None = None,
        description: str | None = None,
        input_schema: Any | None = None,
        output_schema: Any | None = None,
        timeout: float | None = None,
        retries: int = 0,
        retry_delay: float = 1.0,
        requires: list[str] | None = None,
        provides: list[str] | None = None,
        enabled: bool = True,
    ) -> "Workflow":
        """Add a step to the workflow.

        Args:
            func: The async function to execute
            name: Name of the step (defaults to function name)
            description: Description of what the step does
            input_schema: Expected input schema (Pydantic model)
            output_schema: Expected output schema (Pydantic model)
            timeout: Maximum execution time in seconds
            retries: Number of retry attempts on failure
            retry_delay: Delay between retries in seconds
            requires: List of step names that must complete before this step
            provides: List of output keys this step provides
            enabled: Whether the step is enabled

        Returns:
            The workflow instance (for method chaining)
        """
        step_metadata = getattr(func, "_step_metadata", {})

        step = WorkflowStep(
            func=func,
            name=name or step_metadata.get("name", func.__name__),
            description=description or step_metadata.get("description", ""),
            input_schema=input_schema or step_metadata.get("input_schema"),
            output_schema=output_schema or step_metadata.get("output_schema"),
            timeout=timeout if timeout is not None else step_metadata.get("timeout"),
            retries=retries if retries is not None else step_metadata.get("retries", 0),
            retry_delay=retry_delay
            if retry_delay is not None
            else step_metadata.get("retry_delay", 1.0),
            requires=requires or step_metadata.get("requires", []),
            provides=provides or step_metadata.get("provides", []),
            enabled=enabled if enabled is not None else step_metadata.get("enabled", True),
        )

        self.steps.append(step)
        return self

    async def run(
        self,
        initial_data: dict[str, Any] | None = None,
        context: WorkflowContext | None = None,
    ) -> WorkflowContext:
        """Execute the workflow.

        Args:
            initial_data: Initial data for the workflow
            context: Optional existing workflow context

        Returns:
            The workflow context with execution results
        """
        # Initialize context
        if context is None:
            context = WorkflowContext()

        if initial_data:
            context.data.update(initial_data)

        context.status = WorkflowStatus.RUNNING
        context.start_time = datetime.now(UTC)

        self.logger.info(f"Starting workflow '{self.name}'")

        try:
            # Execute steps in order, respecting dependencies
            for step in self.steps:
                if not step.enabled:
                    self.logger.debug(f"Skipping disabled step: {step.name}")
                    context.steps_skipped.append(step.name)
                    continue

                # Check dependencies
                missing_deps = [dep for dep in step.requires if dep not in context.steps_completed]
                if missing_deps:
                    error_msg = (
                        f"Step '{step.name}' is missing dependencies: {', '.join(missing_deps)}"
                    )
                    if self.on_error == "stop":
                        raise StepDependencyError(error_msg)
                    elif self.on_error == "continue":
                        self.logger.warning(f"{error_msg}, skipping step")
                        context.steps_skipped.append(step.name)
                        continue
                    else:  # 'cancel'
                        context.status = WorkflowStatus.CANCELLED
                        self.logger.warning(f"{error_msg}, cancelling workflow")
                        return context

                # Execute the step
                try:
                    await step.execute(context)
                    context.steps_completed.append(step.name)

                except WorkflowStepError as e:
                    context.steps_failed.append(step.name)
                    context.errors[step.name] = str(e)

                    if self.on_error == "stop":
                        raise
                    elif self.on_error == "continue":
                        self.logger.error(f"Step '{step.name}' failed: {e!s}")
                        continue
                    else:  # 'cancel'
                        context.status = WorkflowStatus.CANCELLED
                        self.logger.error(f"Step '{step.name}' failed, cancelling workflow: {e!s}")
                        return context

            # Update final status
            if context.steps_failed:
                context.status = WorkflowStatus.FAILED
            else:
                context.status = WorkflowStatus.COMPLETED

            self.logger.info(
                f"Workflow '{self.name}' completed with status: {context.status.value}"
            )

        except Exception as e:
            context.status = WorkflowStatus.FAILED
            context.errors["workflow"] = str(e)
            self.logger.error(f"Workflow '{self.name}' failed: {e!s}", exc_info=True)
            raise WorkflowError(f"Workflow failed: {e!s}") from e

        finally:
            context.end_time = datetime.now(UTC)
            if context.start_time and context.end_time:
                context.metadata["duration"] = (
                    context.end_time - context.start_time
                ).total_seconds()

        return context
