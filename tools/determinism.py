# SPDX-License-Identifier: AGPL-3.0-only
# sword-sim by jo29_29. Licensed under the GNU AGPL v3 only with additional terms: see LICENSE and ADDITIONAL_TERMS.
"""Determinism check: the same seeds and actions give the same fights, bit for bit, whatever the thread count.

    python tools/determinism.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import numpy as np
import swordsim as ss


def run(threads):
    out = []
    for seed, kw in ((1, {}), (2, dict(rtt_lo=0, rtt_hi=10, jitter=0.2, spike=0.01)), (3, dict(fixed_start=1.0))):
        env = ss.Sword(32, seed=seed, threads=threads, **kw).reset()
        rng = np.random.default_rng(seed)
        for _ in range(1500):
            a = np.zeros((32, 2, ss.ACT_DIM), np.float32)
            a[..., 0:2] = rng.integers(0, 3, (32, 2, 2))
            a[..., 2:6] = rng.random((32, 2, 4)) < [0.1, 0.7, 0.05, 0.3]
            a[..., 6:8] = rng.normal(0, 0.2, (32, 2, 2))
            env.step(a)
            out.append(np.concatenate([env.obs.ravel(), env.reward.ravel(), env.done.astype(np.float32), env.info.ravel()]))
    return np.stack(out)


a, b = run(1), run(os.cpu_count())
bad = np.nonzero((a != b).any(1))[0]
print("DETERMINISTIC" if len(bad) == 0 else f"DIFFERS from step {bad[0]}", a.shape)
