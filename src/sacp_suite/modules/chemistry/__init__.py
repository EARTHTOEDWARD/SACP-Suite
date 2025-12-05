"""Chemistry modules for ADR–Krebs and research ADR ring experiments."""

from .adr_krebs import KrebsADRConfig, make_krebs_adr  # noqa: F401
from .adr_krebs_plugin import ADRKrebsDynamics  # noqa: F401  (registers plugin)
from .adr_research import ADRRingDynamics  # noqa: F401  (registers plugin)
from .instrumented_adr import compute_clc_proxy  # noqa: F401

__all__ = [
    "ADRKrebsDynamics",
    "ADRRingDynamics",
    "KrebsADRConfig",
    "make_krebs_adr",
    "compute_clc_proxy",
]
