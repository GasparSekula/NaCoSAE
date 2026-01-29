"""Custom exceptions for the dashboard application.

This module defines application-specific exceptions used for error handling
in the experiment results dashboard.
"""


class CorruptedExperimentError(Exception):
    """Exception thrown when the experiment folder is incomplete."""

    pass
