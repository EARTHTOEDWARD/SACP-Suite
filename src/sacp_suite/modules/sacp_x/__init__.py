"""SACP-X dynamics plugins (Lorenz, ADR tiles, etc.)."""

# Import modules for side-effect registration with the plugin registry.
# Keeping aliases avoids unused-import warnings.
from sacp_suite.modules.sacp_x import lorenz63 as _lorenz  # noqa: F401
from sacp_suite.modules.sacp_x import adr_hemostasis as _adr_hemo  # noqa: F401
