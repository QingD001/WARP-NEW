"""Direct paired measurement; no supervised extrapolation to unseen regions."""

import math


def estimate_benefits(features, outcomes, prior_queries=16.0):
    """Shrink signed mean gains toward zero with a fixed pseudo-query count.

    Zero entries for unprobed regions are selection sentinels, not measurements.
    Reports distinguish unknown gains from measured zeros. This regularization
    is not a confidence interval or evidence of statistical significance.
    """
    if not math.isfinite(prior_queries) or prior_queries < 0:
        raise ValueError("prior_queries must be finite and nonnegative")
    estimates = dict.fromkeys(features, 0.0)
    for region_id, outcome in outcomes.items():
        if region_id not in estimates:
            raise ValueError(f"Unknown probe region: {region_id}")
        n = outcome.query_count
        if n < 0 or not math.isfinite(outcome.gain):
            raise ValueError("Probe count and gain must be valid")
        estimates[region_id] = outcome.gain * n / (n + prior_queries) if n else 0.0
    return estimates
