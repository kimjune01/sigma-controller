import math
from unittest.mock import patch

from pid import DistanceBin, SigmaController


def make_histogram(sigma_true: float, n_bins: int = 20, max_distance: float = 5.0,
                   impressions_per_bin: int = 100) -> list[DistanceBin]:
    """Generate a synthetic histogram from a Gaussian decay curve."""
    bins = []
    for i in range(n_bins):
        d = (i + 0.5) * max_distance / n_bins
        rate = math.exp(-d ** 2 / (2 * sigma_true ** 2))
        conversions = int(rate * impressions_per_bin)
        bins.append(DistanceBin(distance=d, impressions=impressions_per_bin, conversions=conversions))
    return bins


def test_sigma_decreases_when_boundary_rate_low():
    """If conversions at the boundary are below target, sigma should tighten."""
    ctrl = SigmaController(target_rate=0.05, sigma=2.0)
    initial_sigma = ctrl.sigma

    # Histogram where conversions drop off sharply — boundary rate is low
    histogram = make_histogram(sigma_true=0.5)

    with patch("pid.time") as mock_time:
        mock_time.monotonic.return_value = 1.0
        ctrl._prev_time = 0.0
        ctrl.update(histogram)

    assert ctrl.sigma < initial_sigma


def test_sigma_increases_when_boundary_rate_high():
    """If conversions at the boundary exceed target, sigma can expand."""
    ctrl = SigmaController(target_rate=0.01, sigma=1.5)
    initial_sigma = ctrl.sigma

    # Histogram where conversions are strong even at distance — boundary rate is high
    histogram = make_histogram(sigma_true=5.0, n_bins=40)

    with patch("pid.time") as mock_time:
        mock_time.monotonic.return_value = 1.0
        ctrl._prev_time = 0.0
        ctrl.update(histogram)

    assert ctrl.sigma > initial_sigma


def test_sigma_stays_in_bounds():
    ctrl = SigmaController(target_rate=0.05, sigma=0.1, sigma_min=0.05, sigma_max=10.0)

    histogram = make_histogram(sigma_true=0.01)

    with patch("pid.time") as mock_time:
        mock_time.monotonic.return_value = 1.0
        ctrl._prev_time = 0.0
        for _ in range(50):
            mock_time.monotonic.return_value += 1.0
            ctrl.update(histogram)

    assert ctrl.sigma >= ctrl.sigma_min
    assert ctrl.sigma <= ctrl.sigma_max


def test_estimate_sigma_from_curve():
    ctrl = SigmaController(target_rate=0.05, sigma=1.0)

    histogram = make_histogram(sigma_true=1.5, n_bins=40, impressions_per_bin=1000)
    estimated = ctrl.estimate_sigma_from_curve(histogram)

    # Should be close to the true sigma (within 50%)
    assert 0.75 < estimated < 2.25


def test_estimate_sigma_empty_histogram():
    ctrl = SigmaController(target_rate=0.05, sigma=1.0)
    estimated = ctrl.estimate_sigma_from_curve([])
    assert estimated == 1.0  # falls back to current sigma


def test_boundary_conversion_rate():
    ctrl = SigmaController(target_rate=0.05, sigma=1.0)

    bins = [
        DistanceBin(distance=0.5, impressions=100, conversions=80),
        DistanceBin(distance=1.0, impressions=100, conversions=30),  # boundary
        DistanceBin(distance=1.5, impressions=100, conversions=5),
    ]

    rate = ctrl._boundary_conversion_rate(bins)
    # Only the bin at d=1.0 is within 20% of sigma=1.0 (range 0.8–1.2)
    assert rate == 0.30


def test_convergence():
    """Sigma should converge toward equilibrium where boundary rate equals target."""
    true_sigma = 1.5
    # target_rate=0.30 → equilibrium where exp(-d^2/(2*1.5^2)) = 0.30
    # d ≈ 1.5 * sqrt(2 * ln(1/0.30)) ≈ 1.5 * 1.55 ≈ 2.33
    # Uses default gains (kp=0.3, ki=0.02, kd=0.08)
    ctrl = SigmaController(target_rate=0.30, sigma=5.0)

    with patch("pid.time") as mock_time:
        t = 0.0
        ctrl._prev_time = t

        for i in range(300):
            t += 1.0
            mock_time.monotonic.return_value = t
            histogram = make_histogram(sigma_true=true_sigma, n_bins=40, impressions_per_bin=500)
            ctrl.update(histogram)

    equilibrium = true_sigma * math.sqrt(-2 * math.log(0.30))
    # Should converge within 15% of equilibrium
    assert abs(ctrl.sigma - equilibrium) / equilibrium < 0.15


def test_convergence_openauction_regime():
    """Sigma converges in the OpenAuction operating range (sigma 0.30-0.55)."""
    # Specialist scenario: tight cluster, sigma=0.40
    true_sigma = 0.40
    target_rate = 0.20
    ctrl = SigmaController(target_rate=target_rate, sigma=1.0)

    with patch("pid.time") as mock_time:
        t = 0.0
        ctrl._prev_time = t

        for i in range(300):
            t += 1.0
            mock_time.monotonic.return_value = t
            histogram = make_histogram(sigma_true=true_sigma, n_bins=40, impressions_per_bin=500)
            ctrl.update(histogram)

    equilibrium = true_sigma * math.sqrt(-2 * math.log(target_rate))
    # Should converge within 10% of equilibrium
    assert abs(ctrl.sigma - equilibrium) / equilibrium < 0.10
