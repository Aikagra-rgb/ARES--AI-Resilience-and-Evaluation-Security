"""ARES OpenTelemetry Tracing Module.

Provides distributed tracing, context propagation, and integration
with Grafana Tempo/Jaeger for observability.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, ContextManager

from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.semconv.resource import OTEL_RESOURCE_ATTRIBUTES
from opentelemetry.trace import Tracer, Status, SpanKind, DefaultSpan
from opentelemetry.context import Context, get_current, set_value, get_value
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.http import HTTPInstrumentor
from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter as OTLPGrpcSpanExporter
from opentelemetry.util.types import Attributes

logger = logging.getLogger(__name__)


@dataclass
class TracingConfig:
    """Configuration for OpenTelemetry tracing."""
    
    # Exporter configuration
    otlp_endpoint: str = "http://localhost:4318/v1/traces"
    otlp_grpc: bool = False
    environment: str = "development"
    
    # Service identification
    service_name: str = "ares-api"
    service_version: str = "0.1.0"
    service_namespace: str = "ares"
    
    # Sampling
    sampling_probability: float = 1.0  # 100% in production, adjust as needed
    sampling_parent: str = "always_on"  # or "traceidratio"
    
    # Resource attributes
    deployment_env: str = "development"
    host_name: str = "unknown"


class TracingProvider(ABC):
    """Abstract base class for tracing providers."""
    
    @abstractmethod
    def init_tracing(self) -> None:
        """Initialize the OpenTelemetry tracer provider."""
        pass
    
    @abstractmethod
    def get_tracer(self, name: str) -> Tracer:
        """Get a tracer for the given name."""
        pass
    
    @abstractmethod
    def start_span(self, name: str, **kwargs) -> ContextManager:
        """Start a new span."""
        pass


class OTLP TracingProvider(TracingProvider):
    """OpenTelemetry tracing provider using OTLP exporter."""
    
    def __init__(self, config: TracingConfig = None):
        self.config = config or TracingConfig()
        self._provider: Optional[TracerProvider] = None
        self._tracer: Optional[Tracer] = None
    
    def init_tracing(self) -> None:
        """Initialize the OpenTelemetry tracer provider with OTLP exporter."""
        # Create resource with service information
        resource = Resource.create({
            OTEL_RESOURCE_ATTRIBUTES.service_name: self.config.service_name,
            OTEL_RESOURCE_ATTRIBUTES.service_version: self.config.service_version,
            OTEL_RESOURCE_ATTRIBUTES.deployment_env: self.config.deployment_env,
            "host.name": self.config.host_name,
            "service.namespace": self.config.service_namespace,
        })
        
        # Create tracer provider
        self._provider = TracerProvider(resource=resource)
        
        # Choose exporter based on configuration
        if self.config.otlp_grpc:
            exporter = OTLPGrpcSpanExporter(endpoint=self.config.otlp_endpoint)
        else:
            exporter = OTLPSpanExporter(endpoint=self.config.otlp_endpoint)
        
        # Add batch span processor
        self._provider.add_span_processor(BatchSpanProcessor(exporter))
        
        # Set as global provider
        trace_api.set_tracer_provider(self._provider)
        
        # Get tracer
        self._tracer = self._provider.get_tracer(
            self.config.service_name,
            self.config.service_version,
        )
        
        logger.info(
            "OpenTelemetry tracing initialized",
            extra={
                "otlp_endpoint": self.config.otlp_endpoint,
                "otlp_grpc": self.config.otlp_grpc,
                "environment": self.config.environment,
            },
        )
    
    def get_tracer(self, name: str) -> Tracer:
        """Get a tracer for the given name."""
        if self._tracer is None:
            self.init_tracing()
        return self._provider.get_tracer(name)
    
    def start_span(self, name: str, **kwargs) -> ContextManager:
        """Start a new span with the given name."""
        if self._tracer is None:
            self.init_tracing()
        return self._tracer.start_span(name, **kwargs)


# Global tracing provider instance
_tracing_provider: Optional[TracingProvider] = None


def get_tracing_provider() -> TracingProvider:
    """Get or create the global tracing provider."""
    global _tracing_provider
    if _tracing_provider is None:
        _tracing_provider = OTLPTracingProvider()
    return _tracing_provider


def init_tracing(config: Optional[TracingConfig] = None) -> TracingProvider:
    """Initialize the global tracing provider."""
    global _tracing_provider
    _tracing_provider = OTLPTracingProvider(config=config)
    _tracing_provider.init_tracing()
    return _tracing_provider


def shutdown_tracing() -> None:
    """Shutdown the global tracing provider."""
    global _tracing_provider
    if _tracing_provider:
        # Wait for any pending exports
        if _tracing_provider._provider:
            # Note: In a real implementation, we'd wait for export to complete
            pass
        _tracing_provider = None


# --- Span utilities ---

def trace_execution(operation: str, record_metrics: bool = True):
    """
    Decorator to trace function execution.
    
    Args:
        operation: Name of the operation being traced
        record_metrics: Whether to record latency metrics
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            tracer = get_tracing_provider().get_tracer(__name__)
            with tracer.start_as_current_span(
                operation,
                kind=SpanKind.INTERNAL,
                attributes={
                    "function.name": func.__name__,
                    "module.name": func.__module__,
                },
            ) as span:
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    span.set_status(Status.Ok)
                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status.Error with message=str(e))
                    raise
                finally:
                    duration = time.time() - start_time
                    if record_metrics and span.is_recording():
                        span.set_attribute("duration_ms", round(duration * 1000, 2))
                        span.set_attribute("operation", operation)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            tracer = get_tracing_provider().get_tracer(__name__)
            with tracer.start_as_current_span(
                operation,
                kind=SpanKind.INTERNAL,
                attributes={
                    "function.name": func.__name__,
                    "module.name": func.__module__,
                },
            ) as span:
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    span.set_status(Status.Ok)
                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status.Error with message=str(e))
                    raise
                finally:
                    duration = time.time() - start_time
                    if record_metrics and span.is_recording():
                        span.set_attribute("duration_ms", round(duration * 1000, 2))
                        span.set_attribute("operation", operation)
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper


def trace_http_request(skip_successful: bool = True):
    """
    Decorator/ middleware to trace HTTP requests.
    
    This is typically used with FastAPIInstrumentor, but can also
    be used as a decorator on route functions.
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            tracer = get_tracing_provider().get_tracer(__name__)
            with tracer.start_as_current_span(
                "http.route",
                kind=SpanKind.INTERNAL,
                attributes={
                    "http.method": getattr(args[0], 'method', 'UNKNOWN') if args else 'UNKNOWN',
                    "http.path": getattr(args[0], 'url', 'UNKNOWN').path if args else 'UNKNOWN',
                },
            ) as span:
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    if span.is_recording():
                        span.set_attribute("http.status_code", getattr(result, 'status_code', 200))
        return wrapper
    return decorator


# --- Context propagation ---

class TraceContext:
    """Utilities for trace context propagation."""
    
    @staticmethod
    def extract_trace_context(headers: Dict[str, str]) -> Optional[Context]:
        """Extract trace context from HTTP headers."""
        import base64
        
        trace_id = headers.get("X-Trace-Id")
        span_id = headers.get("X-Span-Id")
        trace_flags = headers.get("X-Trace-Flags", "1")
        
        if not trace_id or not span_id:
            return None
        
        # Build context from traceparent header format
        traceparent = headers.get("Traceparent")
        if traceparent:
            # Format: "traceid-parentid-flags-tracecontext"
            parts = traceparent.split("-")
            if len(parts) == 4:
                trace_id = parts[0]
                span_id = parts[1]
                trace_flags = parts[3] if len(parts) > 3 else "1"
        
        # Create a minimal context
        try:
            ctx = Context()
            ctx = set_value(ctx, "trace_id", trace_id, value_type=str)
            ctx = set_value(ctx, "span_id", span_id, value_type=str)
            ctx = set_value(ctx, "trace_flags", trace_flags, value_type=str)
            return ctx
        except Exception:
            return None
    
    @staticmethod
    def inject_trace_context(headers: Dict[str, str]) -> Dict[str, str]:
        """Inject trace context into HTTP headers."""
        # This would normally use the W3C tracecontext format
        # For now, we just add placeholder headers
        return headers


# --- Convenience function ---

def start_span_from_request(
    request,
    operation: str,
    attributes: Dict[str, Any] = None,
) -> tuple:
    """
    Start a span from a FastAPI request object.
    
    Returns:
        Tuple of (span, context)
    """
    tracer = get_tracing_provider().get_tracer(__name__)
    
    # Extract existing trace context from headers
    trace_header = getattr(request, 'headers', {}).get("X-Trace-Id", "")
    span_header = getattr(request, 'headers', {}).get("X-Span-Id", "")
    
    if trace_header and span_header:
        # Continue existing span
        span = tracer.start_span(
            operation,
            context=Context(),
            kind=SpanKind.INTERNAL,
        )
    else:
        # Start new span
        span = tracer.start_span(
            operation,
            kind=SpanKind.INTERNAL,
            attributes=(attributes or {}),
        )
    
    return span


# Initialize on import - will be configured by the app
_tracer: Optional[Tracer] = None


def get_tracer() -> Tracer:
    """Get the global tracer, initializing if needed."""
    global _tracer
    if _tracer is None:
        from ares.config import settings as default_settings
        provider = get_tracing_provider()
        _tracer = provider.get_tracer(
            default_settings.service_name or "ares-api",
            default_settings.service_version or "0.1.0",
        )
    return _tracer


# Export convenience
__all__ = [
    "TracingConfig",
    "TracingProvider",
    "OTLP TracingProvider",
    "get_tracing_provider",
    "init_tracing",
    "shutdown_tracing",
    "trace_execution",
    "trace_http_request",
    "TraceContext",
    "start_span_from_request",
    "get_tracer",
]