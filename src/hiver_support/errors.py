class ConfigurationError(RuntimeError):
    """Configuration is absent or internally inconsistent."""


class HumanApprovalRequired(ConfigurationError):
    """A required human ground-truth decision has not been recorded."""


class ProviderUnavailable(RuntimeError):
    """A configured model provider cannot be reached or authenticated."""

