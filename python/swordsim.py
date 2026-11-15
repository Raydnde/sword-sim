# SPDX-License-Identifier: AGPL-3.0-only
# sword-sim by jo29_29. Licensed under the GNU AGPL v3 only with additional terms: see LICENSE and ADDITIONAL_TERMS.
"""ctypes wrapper around the Rust sword fights (libswordsim.so, src/env.rs).

The buffers the env returns are its own and are overwritten by the next step: copy what you keep.
"""
import ctypes as C
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.environ.get("SWORD_LIB", os.path.join(HERE, "..", "target", "release", "libswordsim.so"))


class SwordConfig(C.Structure):
    _fields_ = [
        ("max_ticks", C.c_uint32),
        ("rtt_lo", C.c_uint32),
        ("rtt_hi", C.c_uint32),
        ("jitter", C.c_float),
        ("spike", C.c_float),
        ("dist_lo", C.c_float),
        ("dist_hi", C.c_float),
        ("fixed_start", C.c_float),
    ]
    # Mirrors SwordConfig::default() in src/env.rs.
    defaults = dict(max_ticks=3600, rtt_lo=0, rtt_hi=6, jitter=0.05, spike=0.002, dist_lo=2.5, dist_hi=16.0, fixed_start=0.5)


_lib = C.CDLL(LIB)
_f32p = np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")
_u8p = np.ctypeslib.ndpointer(np.uint8, flags="C_CONTIGUOUS")
_lib.sword_new.restype = C.c_void_p
_lib.sword_new.argtypes = [C.c_size_t, C.c_uint64, C.c_size_t]
_lib.sword_free.argtypes = [C.c_void_p]
_lib.sword_configure.argtypes = [C.c_void_p, SwordConfig]
_lib.sword_reset.argtypes = [C.c_void_p]
_lib.sword_observe.argtypes = [C.c_void_p, _f32p, _f32p, _f32p]
_lib.sword_step.argtypes = [C.c_void_p, _f32p, _f32p, _f32p, _f32p, _f32p, _u8p, _f32p]
_lib.sword_state.argtypes = [C.c_void_p, C.c_size_t, np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")]
_dims = (C.c_size_t * 7)()
_lib.sword_dims(_dims)
OBS_DIM, PRIV_DIM, ACT_DIM, INFO_DIM, SELF_DIM, OPP_DIM, N_HIST = _dims

# The action (ACT_DIM floats): fwd, strafe, jump, sprint, sneak, attack, yaw, pitch.
A_FWD, A_STRAFE, A_JUMP, A_SPRINT, A_SNEAK, A_ATTACK, A_YAW, A_PITCH = range(8)
# info: outcome (0/1: that side won, 2: timeout), ticks, rtt of 0 and 1, damage dealt by 0 and 1, hits landed by 0 and 1.
INFO_KEYS = ["outcome", "ticks", "rtt0", "rtt1", "dealt0", "dealt1", "hits0", "hits1"]
WIN0, WIN1, TIMEOUT = 0, 1, 2


class Sword:
    """n fights of two sides. Arrays are (n, 2, ...) per side, done and info (n, ...)."""

    def __init__(self, n, seed=0, threads=None, **cfg):
        self.n = n
        self._s = _lib.sword_new(n, seed, threads or os.cpu_count())
        self.obs = np.zeros((n, 2, OBS_DIM), np.float32)
        self.priv = np.zeros((n, 2, PRIV_DIM), np.float32)
        self.phi = np.zeros((n, 2), np.float32)
        self.reward = np.zeros((n, 2), np.float32)
        self.done = np.zeros(n, np.uint8)
        self.info = np.zeros((n, INFO_DIM), np.float32)
        self.cfg = dict(SwordConfig.defaults)
        self.configure(**cfg)

    def __del__(self):
        if getattr(self, "_s", None):
            _lib.sword_free(self._s)
            self._s = None

    def configure(self, **kw):
        unknown = set(kw) - set(SwordConfig.defaults)
        assert not unknown, unknown
        self.cfg.update(kw)
        _lib.sword_configure(self._s, SwordConfig(**self.cfg))

    def reset(self):
        """New fights everywhere; returns the observation buffers."""
        _lib.sword_reset(self._s)
        _lib.sword_observe(self._s, self.obs, self.priv, self.phi)
        return self

    def step(self, actions):
        _lib.sword_step(self._s, np.ascontiguousarray(actions, np.float32).reshape(self.n * 2 * ACT_DIM), self.obs,
                        self.priv, self.phi, self.reward, self.done, self.info)
        return self

    def state(self, k):
        """Slot k for a viewer: per side x, y, z, yaw, pitch, health, absorption, hurt time; then the tick."""
        out = np.zeros(17)
        _lib.sword_state(self._s, k, out)
        return out
