"""Basic chaos-band controller."""

from __future__ import annotations

from dataclasses import dataclass

from sacp_suite.selftuner.types import (
    ChaosBand,
    LyapunovEstimate,
    TuningAction,
    TuningContext,
)


@dataclass
class SimpleChaosBandController:
    """Proportional controller that keeps λ within a band."""

    target_band: ChaosBand
    k_spectral_radius: float = 0.1
    k_gain: float = 0.0

    def tune(self, estimate: LyapunovEstimate, context: TuningContext) -> TuningAction:
        if estimate.values.size == 0:
            return TuningAction(metadata={"regime": "unknown"})
        lam = float(estimate.values[0])
        regime = "within_band"
        delta_sr = 0.0
        if lam < self.target_band.lower:
            regime = "too_ordered"
            delta_sr = self.k_spectral_radius
        elif lam > self.target_band.upper:
            regime = "too_chaotic"
            delta_sr = -self.k_spectral_radius

        return TuningAction(
            spectral_radius_delta=delta_sr,
            gain_delta=self.k_gain,
            metadata={
                "regime": regime,
                "lambda_max": lam,
                "system_id": context.system_id,
            },
        )
