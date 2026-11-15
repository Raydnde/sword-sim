# SPDX-License-Identifier: AGPL-3.0-only
# sword-sim by jo29_29. Licensed under the GNU AGPL v3 only with additional terms: see LICENSE and ADDITIONAL_TERMS.
"""Both sides of 256 duels play random actions until every duel has ended once; prints the outcomes.

    cargo build --release && python examples/random_policy.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import numpy as np
import swordsim as ss

n = 256
env = ss.Sword(n, seed=7).reset()
rng = np.random.default_rng(0)
ended = np.zeros(n, bool)
result = np.zeros(n, int)
while not ended.all():
    a = np.zeros((n, 2, ss.ACT_DIM), np.float32)
    a[..., ss.A_FWD] = 2                                   # always forward
    a[..., ss.A_STRAFE] = rng.integers(0, 3, (n, 2))
    a[..., ss.A_JUMP:ss.A_ATTACK + 1] = rng.random((n, 2, 4)) < [0.1, 0.9, 0.0, 0.2]
    a[..., ss.A_YAW:] = rng.normal(0, 0.02, (n, 2, 2))     # roughly at the opponent
    env.step(a)
    for k in np.nonzero(env.done.astype(bool) & ~ended)[0]:
        ended[k] = True
        result[k] = int(env.info[k, 0])
print(f"side 0 won {np.mean(result == 0):.1%}, side 1 won {np.mean(result == 1):.1%}, timeouts {np.mean(result == 2):.1%}")
