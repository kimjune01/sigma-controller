"""
PID controller for sigma — the advertiser's reach parameter.

The advertiser sets a margin. The exchange observes conversion rates at different
distances from the advertiser's center via distance histograms and verified
conversions. The controller adjusts sigma to maximize conversions within the
margin the advertiser can sustain.
"""

import time
from dataclasses import dataclass, field


@dataclass
class DistanceBin:
    """A single bin from a distance histogram."""

    distance: float       # center of the bin
    impressions: int      # number of impressions in this bin
    conversions: int      # verified conversions in this bin

    @property
    def conversion_rate(self) -> float:
        if self.impressions == 0:
            return 0.0
        return self.conversions / self.impressions


@dataclass
class SigmaController:
    """PID controller that adjusts sigma to hit a target conversion rate at the boundary."""

    target_rate: float    # conversion rate the advertiser's margin can sustain
    kp: float = 0.3      # proportional gain
    ki: float = 0.02     # integral gain
    kd: float = 0.08     # derivative gain
    sigma: float = 1.0   # initial sigma
    sigma_min: float = 0.05
    sigma_max: float = 10.0

    _integral: float = field(default=0.0, init=False)
    _prev_error: float = field(default=0.0, init=False)
    _prev_time: float = field(default=0.0, init=False)

    def __post_init__(self):
        self._prev_time = time.monotonic()

    def update(self, histogram: list[DistanceBin]) -> float:
        """
        Update sigma based on a distance histogram with verified conversions.

        The boundary is defined as the bins near the current sigma. If the
        conversion rate at the boundary is below the target, sigma is too wide.
        If above, sigma could expand.

        Args:
            histogram: list of DistanceBin, sorted by distance

        Returns:
            The new sigma value.
        """
        now = time.monotonic()
        dt = now - self._prev_time
        if dt <= 0:
            return self.sigma

        boundary_rate = self._boundary_conversion_rate(histogram)

        # Error: positive means boundary rate is below target — sigma is too wide
        error = self.target_rate - boundary_rate

        self._integral += error * dt

        derivative = (error - self._prev_error) / dt

        adjustment = (self.kp * error) + (self.ki * self._integral) + (self.kd * derivative)

        # Positive adjustment = sigma too wide, pull it in (decrease)
        self.sigma -= adjustment
        self.sigma = max(self.sigma_min, min(self.sigma_max, self.sigma))

        self._prev_error = error
        self._prev_time = now

        return self.sigma

    def _boundary_conversion_rate(self, histogram: list[DistanceBin]) -> float:
        """
        Compute the conversion rate at the boundary (bins near current sigma).

        Uses bins within 20% of sigma as the boundary region.
        """
        margin = max(0.3 * self.sigma, 0.25)
        lower = self.sigma - margin
        upper = self.sigma + margin

        total_impressions = 0
        total_conversions = 0

        for bin in histogram:
            if lower <= bin.distance <= upper:
                total_impressions += bin.impressions
                total_conversions += bin.conversions

        if total_impressions == 0:
            return 0.0

        return total_conversions / total_impressions

    def estimate_sigma_from_curve(self, histogram: list[DistanceBin]) -> float:
        """
        Estimate sigma from the Gaussian decay curve in the histogram.

        Fits conversion_rate ~ exp(-d^2 / (2 * sigma^2)) by finding the
        distance at which the conversion rate drops to 1/e of the peak.

        This is used for initial sigma estimation before the PID loop takes over.
        """
        if not histogram:
            return self.sigma

        peak_rate = max(b.conversion_rate for b in histogram)
        if peak_rate == 0:
            return self.sigma

        import math
        threshold = peak_rate / math.e

        sorted_bins = sorted(histogram, key=lambda b: b.distance)

        for bin in sorted_bins:
            if bin.conversion_rate <= threshold and bin.impressions > 0:
                return max(self.sigma_min, bin.distance)

        # If no bin drops below threshold, sigma is at least as wide as the histogram
        return sorted_bins[-1].distance if sorted_bins else self.sigma
