# sword-sim

A headless, tick-exact Minecraft 1.21.11 sword-duel simulator for reinforcement learning. It covers the server,
both clients with their own prediction, and the network lag between them, with thousands of duels batched across
threads. It exposes a C ABI and ships a small Python (ctypes + numpy) wrapper.

Sword only: diamond armor (Protection III, Unbreaking III), a diamond sword (Unbreaking III) and vanilla food
(20 food, 5 saturation, natural regeneration, Normal difficulty). The arena is a walled oval floor, 67 by 65 blocks,
all bedrock. Nothing else is in the game: no other items, effects, blocks,
fluids or projectiles.

## What is simulated

- **Movement.** Player movement and collision ported line for line from the game (LocalPlayer.aiStep,
  LivingEntity.travel, Entity.move), keeping Java's float/double order and widths and Mth's sine table.
- **Combat.** Player.attack: charge scaling, crits, sprint knockback, sword durability and exhaustion.
- **Damage.** The hurt pipeline: invulnerability ticks, knockback, armor and toughness, Protection, absorption,
  fall damage and suffocation; vanilla FoodData (exhaustion, saturation, regeneration, starvation).
- **Client and server.** Each client predicts locally and sees the other player through interpolation. The server
  echoes moves and runs its own no-input physics for each player. Packets go in the game's order within a tick.
- **Lag.** Per-side ping in whole ticks, with jitter and spikes, and in-order (TCP-like) delivery.

## Layout

    src/
      lib.rs           crate root
      sim/
        world.rs       vectors, collision, player movement (Body)
        server.rs      the server: players, items, damage, attacks, packets
        client.rs      the clients and the order of one tick (Duel)
      fight.rs         one fight: kit, arena, ping, start positions, end conditions
      env.rs           batched fights, observations, rewards, C ABI
      arena.txt        the arena floor map
    python/swordsim.py ctypes wrapper
    examples/          random_policy.py
    tools/determinism.py  determinism check
    LICENSE            GNU AGPL v3 text
    ADDITIONAL_TERMS   copyright and the section 7 additional terms

## Build

    cargo build --release          # -> target/release/libswordsim.so

## Use

```python
import sys; sys.path.insert(0, "python")
import numpy as np, swordsim as ss

env = ss.Sword(64, seed=1).reset()             # 64 duels, two sides each, both played by you
a = np.zeros((64, 2, ss.ACT_DIM), np.float32)
env.step(a)    # env.obs (64, 2, OBS_DIM), env.priv, env.reward (64, 2), env.done (64,), env.info (64, 8)
```

Each side's action has 8 values:

| Index | Action | Values |
|---|---|---|
| 0 | fwd | 0 back, 1 none, 2 forward |
| 1 | strafe | 0 left, 1 none, 2 right |
| 2–5 | jump, sprint, sneak, attack | — |
| 6 | yaw offset from the line to the opponent's middle | −1..1 = ±180° |
| 7 | pitch offset from that line | −1..1 = ±90° |

The observations are what that side's client knows, in a frame turned toward the opponent, plus its last 4 actions.
`priv` is the server's truth, meant for a critic. The reward is +1/−1 on the fight's last step and 0 on a timeout.

`info` holds the outcome (0 or 1 for the winning side, 2 for a timeout), the ticks, the ping, the damage dealt and
the hits landed. There are no built-in opponents: both sides take actions from outside. Fights reset on their own.
The arrays are reused every step, so copy what you keep. `SWORD_LIB` overrides the library path.

    python examples/random_policy.py

## Configuration

Pass these to `Sword(n, seed, **cfg)` or to `env.configure(**cfg)` later. They apply to every fight that starts after.

| Option | Default | Meaning |
|---|---|---|
| `max_ticks` | 3600 | time limit (20 ticks = 1 s); a fight that reaches it is a draw |
| `rtt_lo`, `rtt_hi` | 0, 6 | each side's round-trip ping in ticks (1 tick = 50 ms), drawn per fight from this range and split between up and down |
| `jitter` | 0.05 | chance per tick that each direction's delay moves to its base −1, 0 or +1 tick |
| `spike` | 0.002 | chance per tick of a lag spike: 2–8 extra ticks for 5–40 ticks |
| `dist_lo`, `dist_hi` | 2.5, 16 | distance between the players at a random start |
| `fixed_start` | 0.5 | chance of the fixed start (at the two ends of the arena, 66 blocks apart, facing each other) instead of a random one |

Lag is per side: each player has its own ping to the server, and packets in each direction arrive in order, as on TCP.
Its client sees the opponent and its own hits late by that much. `rtt_lo = rtt_hi = 0`, `jitter = 0`, `spike = 0`
turns lag off. Each side's base ping is in its observation, both sides' current ping is in `priv`, and the base
round trips are in `info`.

## Determinism

Fights are a pure function of the seed and the actions, whatever the thread count. The check runs 3 seeds × 1500
ticks × 32 duels with random actions, lag included, once on one thread and once on all of them, and compares them bit
for bit:

    python tools/determinism.py     # -> DETERMINISTIC

## License

sword-sim by jo29_29. Copyright (C) 2026 jo29_29.

GNU Affero General Public License v3.0 only (no "or later"), see LICENSE, with additional terms under its
section 7, see ADDITIONAL_TERMS. In short:

- Anyone who distributes this code or a modified version, or runs a modified version as a network service, must
  publish the complete source under the same license and the same additional terms.
- Keep the copyright notice and the attribution "sword-sim by jo29_29" in the source, and show
  "Uses sword-sim by jo29_29" in the legal notices (about or credits screen) of any work that uses it.
- Mark modified versions as changed; don't present them as the original.
- The name "jo29_29" may not be used to promote derived works.
- ADDITIONAL_TERMS must be kept and passed on with every copy.

The mechanics are reimplemented from reading the game's decompiled code; no Minecraft code or assets are included.
Minecraft is a trademark of Mojang/Microsoft; this project is not affiliated with them.
