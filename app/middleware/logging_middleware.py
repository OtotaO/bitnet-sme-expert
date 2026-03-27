"""Logging middleware for FastAPI."""
import time
import logging
from typing import Callable, Awaitable
from fastapi import Request, Response

logger = logging.getLogger(__name__)

class LoggingMiddleware:
    """Middleware for logging HTTP requests and responses."""

    async def __call__(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Log request and response details."""
        # Log request
        start_time = time.time()
        
        # Get client info
        client_host = request.client.host if request.client else "unknown"
        method = request.method
        url = request.url.path
        query_params = request.url.query
        
        logger.info(
            f"Request: {method} {url}?{query_params} from {client_host}"
        )
        
        # Process request
        try:
            response = await call_next(request)
        except Exception as e:
            logger.exception(f"Request failed: {str(e)}")
            raise
            
        # Calculate process time
        process_time = (time.time() - start_time) * 1000
        process_time = round(process_time, 2)
        
        # Log response
        logger.info(
            f"Response: {method} {url}?{query_params} "
            f"status={response.status_code} "
            f"took={process_time}ms"
        )
        
        return response
