"""
Phoenix/OpenTelemetry tracing integration for the Plan Generator workflow.

Provides tracing with Arize Phoenix for observability.
"""

from contextlib import contextmanager
from typing import Optional, Any

from opentelemetry import trace
from opentelemetry.trace import Tracer, TracerProvider

# OpenInference semantic convention
OPENINFERENCE_SPAN_KIND = "openinference.span.kind"

# Global state for tracing
_tracer: Optional[Tracer] = None
_tracing_enabled: bool = False


def setup_tracing(
    enabled: bool = True,
    project_name: str = "plan-generator",
    endpoint: str = "http://localhost:6006/v1/traces"
) -> Optional[TracerProvider]:
    """
    Initialize Phoenix tracing.

    Args:
        enabled: Whether to enable tracing
        project_name: Phoenix project name
        endpoint: Phoenix OTLP endpoint

    Returns:
        TracerProvider if tracing is enabled, None otherwise.
    """
    global _tracer, _tracing_enabled

    if not enabled:
        _tracing_enabled = False
        return None

    try:
        from phoenix.otel import register

        tracer_provider = register(
            project_name=project_name,
            endpoint=endpoint,
        )

        _tracer = trace.get_tracer("plan-generator-workflow")
        _tracing_enabled = True

        print(f"Phoenix tracing enabled: {endpoint} (project: {project_name})")
        return tracer_provider

    except ImportError as e:
        print(f"Warning: Tracing dependencies not installed: {e}")
        print("Install with: pip install arize-phoenix opentelemetry-sdk opentelemetry-exporter-otlp")
        _tracing_enabled = False
        return None
    except Exception as e:
        print(f"Warning: Failed to initialize tracing: {e}")
        _tracing_enabled = False
        return None


def get_tracer() -> Optional[Tracer]:
    """Get the configured tracer, or None if tracing is disabled."""
    return _tracer if _tracing_enabled else None


def is_tracing_enabled() -> bool:
    """Check if tracing is currently enabled."""
    return _tracing_enabled


@contextmanager
def workflow_span(name: str, **attributes: Any):
    """
    Context manager for creating a workflow-level span.

    Args:
        name: Name of the span (e.g., "analyze-paper-workflow")
        **attributes: Additional span attributes
    """
    if not _tracing_enabled or _tracer is None:
        yield None
        return

    with _tracer.start_as_current_span(name) as span:
        span.set_attribute(OPENINFERENCE_SPAN_KIND, "WORKFLOW")
        for key, value in attributes.items():
            if value is not None:
                span.set_attribute(key, str(value) if not isinstance(value, (int, float, bool)) else value)
        yield span
