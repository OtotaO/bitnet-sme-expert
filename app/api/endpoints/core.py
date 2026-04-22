from fastapi import APIRouter, Depends, HTTPException, status, Query, Body, Path
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
import logging
import uuid
from datetime import datetime

from app.services.expert_service import ExpertService, get_expert_service
from app.schemas.base import ExpertDomain, BaseResponse, ErrorResponse
from app.schemas.request import QueryRequest, CollaborateRequest, TrainingRequest, FeedbackRequest, SearchRequest
from app.schemas.response import (
    QueryResponse,
    CollaborateResponse,
    TrainingJobResponse,
    ListExpertsResponse,
    ExpertInfo,
    ExpertResponse
)
from app.core.workflow import Workflow, WorkflowContext, WorkflowStatus

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get(
    "/health",
    response_model=BaseResponse,
    summary="Health check",
    description="Check if the API is running",
    tags=["System"]
)
async def health_check():
    """Health check endpoint."""
    return BaseResponse(
        success=True,
        message="API is running",
        timestamp=datetime.utcnow()
    )

@router.get(
    "/experts",
    response_model=ListExpertsResponse,
    summary="List available experts",
    description="Get a list of all available experts with their capabilities",
    tags=["Experts"]
)
async def list_experts(
    expert_service: ExpertService = Depends(get_expert_service)
):
    """List all available experts."""
    try:
        experts = await expert_service.list_experts()
        return ListExpertsResponse(
            success=True,
            message=f"Found {len(experts)} experts",
            count=len(experts),
            data=experts,
            timestamp=datetime.utcnow()
        )
    except Exception as e:
        logger.error(f"Error listing experts: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing experts: {str(e)}"
        )

@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Query an expert",
    description="Ask a question to a specific expert or auto-route to the best expert",
    tags=["Query"]
)
async def query_expert(
    request: QueryRequest,
    expert_service: ExpertService = Depends(get_expert_service)
):
    """Query an expert with a question."""
    query_id = f"qry_{uuid.uuid4().hex[:12]}"
    
    try:
        # Start timing
        start_time = datetime.utcnow()
        
        # Create a workflow for this query
        workflow = Workflow(
            name="query_expert",
            description="Process a query to an expert",
            on_error="stop"
        )
        
        # Define steps
        @workflow.step(
            name="validate_input",
            description="Validate the input parameters"
        )
        async def validate_input(context: WorkflowContext):
            if not request.question.strip():
                raise ValueError("Question cannot be empty")
            if request.max_tokens < 1 or request.max_tokens > 4096:
                raise ValueError("max_tokens must be between 1 and 4096")
            return {"is_valid": True}
        
        @workflow.step(
            name="select_expert",
            description="Select the appropriate expert for the query",
            requires=["validate_input"]
        )
        async def select_expert(context: WorkflowContext):
            if request.domain:
                # Use the specified domain
                domain = request.domain
                expert = await expert_service.get_experts_by_domain(domain.value)
                if not expert:
                    raise ValueError(f"No experts available for domain: {domain.value}")
                expert = expert[0]  # Use the first available expert for the domain
            else:
                # Auto-route to the best expert based on the question
                # This is a simple implementation - in practice, you might use a more sophisticated routing strategy
                question = request.question.lower()
                
                # Simple keyword-based routing
                if any(term in question for term in ["math", "calculate", "equation"]):
                    domain = ExpertDomain.MATH
                elif any(term in question for term in ["code", "program", "algorithm"]):
                    domain = ExpertDomain.CODE
                else:
                    domain = ExpertDomain.GENERAL
                
                expert = await expert_service.get_experts_by_domain(domain.value)
                if not expert:
                    # Fall back to any available expert
                    all_experts = await expert_service.get_all_experts()
                    if not all_experts:
                        raise ValueError("No experts available")
                    expert = list(all_experts.values())[0]
                else:
                    expert = expert[0]
            
            return {
                "selected_expert_id": expert.id,
                "expert_name": expert.name,
                "expert_domain": expert.domain.value
            }
        
        @workflow.step(
            name="generate_response",
            description="Generate a response using the selected expert",
            requires=["select_expert"]
        )
        async def generate_response(context: WorkflowContext):
            expert_id = context.data["selected_expert_id"]
            response = await expert_service.query_expert(expert_id, request)
            return {"expert_response": response}
        
        # Run the workflow
        context = await workflow.run()
        
        # Check if the workflow completed successfully
        if context.status != WorkflowStatus.COMPLETED:
            error_msg = context.errors.get("workflow", "Unknown error")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to process query: {error_msg}"
            )
        
        # Extract the response
        expert_response = context.data["expert_response"]
        
        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Prepare the response
        return QueryResponse(
            success=True,
            message="Query processed successfully",
            timestamp=datetime.utcnow(),
            query_id=query_id,
            data=ExpertResponse(
                expert_id=context.data["selected_expert_id"],
                expert_name=context.data["expert_name"],
                domain=ExpertDomain(context.data["expert_domain"]),
                response=expert_response.get("response", ""),
                confidence=expert_response.get("confidence", 1.0),
                model=expert_response.get("model", "unknown"),
                tokens_used=expert_response.get("tokens_used", 0),
                processing_time=processing_time,
                metadata=expert_response.get("metadata", {}),
                sources=expert_response.get("sources", [])
            )
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )

@router.post(
    "/collaborate",
    response_model=CollaborateResponse,
    summary="Collaborate with multiple experts",
    description="Get responses from multiple experts and optionally combine them",
    tags=["Collaboration"]
)
async def collaborate(
    request: CollaborateRequest,
    expert_service: ExpertService = Depends(get_expert_service)
):
    """Collaborate with multiple experts on a question."""
    query_id = f"col_{uuid.uuid4().hex[:12]}"
    
    try:
        # Start timing
        start_time = datetime.utcnow()
        
        # Get responses from all specified experts
        responses = await expert_service.collaborate(request)
        
        # Process responses
        expert_responses = {}
        for expert_id, response in responses.items():
            if response.get("success", False):
                expert = await expert_service.get_expert(expert_id)
                expert_responses[expert_id] = ExpertResponse(
                    expert_id=expert_id,
                    expert_name=expert.name,
                    domain=expert.domain,
                    response=response.get("response", ""),
                    confidence=response.get("confidence", 1.0),
                    model=response.get("model", "unknown"),
                    tokens_used=response.get("tokens_used", 0),
                    processing_time=response.get("processing_time", 0),
                    metadata=response.get("metadata", {}),
                    sources=response.get("sources", [])
                )
        
        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        return CollaborateResponse(
            success=True,
            message=f"Collaboration completed with {len(expert_responses)} experts",
            timestamp=datetime.utcnow(),
            query_id=query_id,
            data=expert_responses,
            metadata={
                "processing_time": processing_time,
                "total_experts_queried": len(responses),
                "successful_responses": len(expert_responses)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in collaboration: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in collaboration: {str(e)}"
        )

@router.post(
    "/train",
    response_model=TrainingJobResponse,
    summary="Train a custom expert",
    description="Start a training job for a new custom expert",
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Training"]
)
async def train_expert(
    request: TrainingRequest,
    expert_service: ExpertService = Depends(get_expert_service)
):
    """Train a new custom expert."""
    try:
        # In a real implementation, this would start an async training job
        # For now, we'll just return a mock response
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        
        return TrainingJobResponse(
            success=True,
            message=f"Training job started for expert '{request.name}'",
            timestamp=datetime.utcnow(),
            job_id=job_id,
            name=request.name,
            status="pending",
            progress=0.0,
            created_at=datetime.utcnow(),
            metadata={
                "domain": request.domain.value,
                "base_model": request.base_model,
                "training_examples": len(request.training_data)
            }
        )
        
    except Exception as e:
        logger.error(f"Error starting training job: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error starting training job: {str(e)}"
        )

@router.post(
    "/feedback",
    response_model=BaseResponse,
    summary="Provide feedback",
    description="Provide feedback on an expert's response",
    tags=["Feedback"]
)
async def submit_feedback(
    request: FeedbackRequest,
    expert_service: ExpertService = Depends(get_expert_service)
):
    """Submit feedback on an expert's response."""
    try:
        # In a real implementation, this would store the feedback
        # For now, we'll just log it
        logger.info(
            f"Received feedback for query {request.query_id}: "
            f"rating={request.rating}, feedback={request.feedback}"
        )
        
        return BaseResponse(
            success=True,
            message="Feedback received, thank you!",
            timestamp=datetime.utcnow(),
            metadata={
                "query_id": request.query_id,
                "rating": request.rating
            }
        )
        
    except Exception as e:
        logger.error(f"Error processing feedback: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing feedback: {str(e)}"
        )

@router.post(
    "/search",
    response_model=BaseResponse,
    summary="Search across experts",
    description="Search for information across all experts",
    tags=["Search"]
)
async def search_experts(
    request: SearchRequest,
    expert_service: ExpertService = Depends(get_expert_service)
):
    """Search across all experts."""
    try:
        # In a real implementation, this would perform a search across experts
        # For now, we'll just return a mock response
        results = []
        
        # Get all experts if no specific domains are provided
        if not request.domains:
            experts = await expert_service.get_all_experts()
        else:
            experts = {}
            for domain in request.domains:
                domain_experts = await expert_service.get_experts_by_domain(domain)
                for expert in domain_experts:
                    experts[expert.id] = expert
        
        # Mock search results
        for expert_id, expert in list(experts.items())[:request.limit]:
            results.append({
                "expert_id": expert_id,
                "expert_name": expert.name,
                "domain": expert.domain.value,
                "snippet": f"Relevant information about '{request.query}' from {expert.name}",
                "confidence": 0.8,  # Mock confidence score
                "metadata": {
                    "model": getattr(expert.config, "model_name", "unknown"),
                    "is_custom": getattr(expert.config, "is_custom", False)
                }
            })
        
        return BaseResponse(
            success=True,
            message=f"Found {len(results)} relevant results",
            timestamp=datetime.utcnow(),
            data={"results": results},
            metadata={
                "query": request.query,
                "domains": [d.value for d in request.domains] if request.domains else "all",
                "limit": request.limit,
                "threshold": request.threshold
            }
        )
        
    except Exception as e:
        logger.error(f"Error performing search: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error performing search: {str(e)}"
        )
