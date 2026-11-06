"""Exception hierarchy. Every failure mode maps to a distinct, catchable type."""

from __future__ import annotations


class NgoPhotoError(Exception):
    """Base class for all project errors."""


class ConfigError(NgoPhotoError):
    """Invalid or missing configuration."""


class RuleConfigError(ConfigError):
    """The rule-set configuration is invalid."""


class DataError(NgoPhotoError):
    """Dataset problem (missing directory, bad manifest, leakage, ...)."""


class LeakageError(DataError):
    """Train/validation/test leakage detected."""


class InvalidImageError(NgoPhotoError):
    """Image missing, unreadable, unsupported, or too large."""


class AuthorizationError(NgoPhotoError):
    """An action requiring explicit authorization (cloud upload, paid job, ...) was not authorized."""


class ProviderError(NgoPhotoError):
    """Cloud/model provider failure."""

    retryable: bool = False


class TransientProviderError(ProviderError):
    """Timeouts, rate limits, 5xx: safe to retry."""

    retryable = True


class ProviderAuthError(ProviderError):
    """Authentication / permission failure. Not retryable."""


class ModelUnavailableError(ProviderError):
    """Requested model / endpoint does not exist or is not enabled."""


class ResponseValidationError(NgoPhotoError):
    """Model output could not be parsed or failed schema / rule-set validation."""


class TuningError(NgoPhotoError):
    """Tuning job preparation, submission or execution failure."""


class ModelArtifactError(NgoPhotoError):
    """Probability model / calibration artifacts missing or invalid."""
