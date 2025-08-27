"""Status tracking and instrumentation for annotation pipeline."""

from functools import wraps
from annotation_config import PipelineStatus

# Global status instance
status = PipelineStatus()


def track(func):
    """Decorator updating :data:`status` with the last function called.

    Purpose
    -------
    Provide lightweight instrumentation by recording each decorated call. This
    avoids invasive logging and keeps overhead minimal for performance.

    Inputs
    ------
    func: Callable
        Function to wrap.

    Outputs
    -------
    Callable
        Wrapped function that updates :data:`status` before executing.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        status.last_function = func.__name__
        return func(*args, **kwargs)

    return wrapper