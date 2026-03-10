# sigma-controller

PID controller for sigma — the advertiser's reach parameter in a [power-diagram ad auction](https://kimjune01.github.io/set-it-and-forget-it).

The advertiser sets a margin. The exchange observes conversion rates at different distances via distance histograms and verified conversions. The controller adjusts sigma to track the boundary where conversions remain profitable.

## How it works

- **Distance histograms.** The publisher reports aggregated conversion counts per distance bin. No individual embeddings, no timestamps.
- **PID feedback loop.** If the conversion rate at the boundary is below the target the advertiser's margin can sustain, sigma tightens. If it's above, sigma expands. Integral and derivative terms handle drift and sudden changes.
- **Initial estimate from data.** `estimate_sigma_from_curve` fits the Gaussian decay curve before the PID loop takes over.

## Usage

```python
from pid import SigmaController, DistanceBin

controller = SigmaController(target_rate=0.05)  # 5% conversion rate at boundary

# Each update cycle, feed in the latest distance histogram
histogram = [
    DistanceBin(distance=0.5, impressions=100, conversions=80),
    DistanceBin(distance=1.0, impressions=100, conversions=30),
    DistanceBin(distance=1.5, impressions=100, conversions=5),
]

new_sigma = controller.update(histogram)
```

## Tests

```
uv run --with pytest pytest test_pid.py -v
```

## License

MIT
