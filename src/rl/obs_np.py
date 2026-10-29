"""numpy observation builder for the Pi runtime — CONTRACT.md §5, parity §6.

One policy step gives `s_raw` (15 values, fixed order):

    0 goal_dx [m]       1 goal_dy [m]      2 sin(psi_err)   3 cos(psi_err)
    4 u [m/s]           5 v [m/s]          6 r [rad/s]
    7 a_prev_surge      8 a_prev_sway      9 a_prev_yaw
    10 wp1_dx [m]       11 wp1_dy [m]      12 wp2_dx [m]    13 wp2_dy [m]
    14 mode (0 waypoint, 1 station keeping)

`build()` normalizes per step, then stacks K=10 samples oldest-first, newest-last
-> 150 values. First sample after `reset()` fills all 10 slots.

Math goes through `common.frames` (goal_in_body, heading_error_rad) so the torch
builder in `envs/` can use the same formulas. Computed in float64, cast once at the
end to float32 (CONTRACT §6). numpy only — no torch on the Pi.
"""

import numpy as np

from src.rl.frames import goal_in_body, heading_error_rad

__all__ = [
    "RAW_DIM",
    "K_HISTORY",
    "OBS_DIM",
    "DEFAULT_OFFSET",
    "DEFAULT_SCALE",
    "DEFAULT_CLIP_NORM",
    "raw_sample",
    "normalize",
    "ObsBuilder",
]

RAW_DIM = 15
K_HISTORY = 10
OBS_DIM = RAW_DIM * K_HISTORY  # 150

# CONTRACT §3.4 defaults (configs/env.yaml obs block), duplicated so the module works
# without a YAML file; callers pass the values that were actually used for training.
DEFAULT_OFFSET = np.zeros(RAW_DIM, dtype=np.float64)
DEFAULT_SCALE = np.array(
    [10.0, 10.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0, 1.0],
    dtype=np.float64,
)
DEFAULT_CLIP_NORM = 5.0


def _as_batch(a, cols):
    """None -> None; else float64 [B, cols]."""
    if a is None:
        return None
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if a.shape[1] != cols:
        raise ValueError(f"expected {cols} columns, got shape {a.shape}")
    return a


def _wp_in_body(wp, x, y, psi):
    """Waypoint [B,2] ENU -> body frame (dx, dy); None -> zeros (CONTRACT §5)."""
    if wp is None:
        return np.zeros((x.shape[0], 2), dtype=np.float64)
    wp = _as_batch(wp, 2)
    if wp.shape[0] != x.shape[0]:
        raise ValueError("waypoint batch size mismatch")
    dx, dy = goal_in_body(wp[:, 0], wp[:, 1], x, y, psi)
    return np.stack([np.asarray(dx, float), np.asarray(dy, float)], axis=1)


def raw_sample(est, goal, prev_action=None, wp1=None, wp2=None):
    """Unnormalized 15-vector(s).

    Args:
        est: [B, 6] state layout of CONTRACT §2 (x, y, psi, u, v, r), estimator only.
        goal: [B, 4] = (x_goal_e, y_goal_n, psi_goal_rad, mode).
        prev_action: [B, 3] last applied action, zeros at reset.
        wp1, wp2: [B, 2] ENU waypoints after the current goal, or None -> zeros.

    Returns:
        float64 [B, 15].
    """
    est = _as_batch(est, 6)
    goal = _as_batch(goal, 4)
    if est.shape[0] != goal.shape[0]:
        raise ValueError("est/goal batch size mismatch")
    B = est.shape[0]
    x, y, psi = est[:, 0], est[:, 1], est[:, 2]

    dx, dy = goal_in_body(goal[:, 0], goal[:, 1], x, y, psi)
    perr = heading_error_rad(psi, goal[:, 2])

    a = _as_batch(prev_action, 3)
    if a is None:
        a = np.zeros((B, 3), dtype=np.float64)
    elif a.shape[0] != B:
        raise ValueError("prev_action batch size mismatch")

    s = np.empty((B, RAW_DIM), dtype=np.float64)
    s[:, 0] = dx
    s[:, 1] = dy
    s[:, 2] = np.sin(perr)
    s[:, 3] = np.cos(perr)
    s[:, 4:7] = est[:, 3:6]
    s[:, 7:10] = a
    s[:, 10:12] = _wp_in_body(wp1, x, y, psi)
    s[:, 12:14] = _wp_in_body(wp2, x, y, psi)
    s[:, 14] = goal[:, 3]
    return s


def normalize(raw, offset=None, scale=None, clip_norm=DEFAULT_CLIP_NORM):
    """(raw - offset) / scale clipped to +/- clip_norm. `scale` is a divisor."""
    offset = DEFAULT_OFFSET if offset is None else np.asarray(offset, dtype=np.float64)
    scale = DEFAULT_SCALE if scale is None else np.asarray(scale, dtype=np.float64)
    return np.clip((np.asarray(raw, dtype=np.float64) - offset) / scale,
                   -clip_norm, clip_norm)


class ObsBuilder:
    """Ring buffer of K normalized samples. One `build()` per policy step."""

    def __init__(self, offset=None, scale=None, clip_norm=DEFAULT_CLIP_NORM,
                 k_history=K_HISTORY):
        self.offset = DEFAULT_OFFSET if offset is None else np.asarray(offset, dtype=np.float64)
        self.scale = DEFAULT_SCALE if scale is None else np.asarray(scale, dtype=np.float64)
        self.clip_norm = float(clip_norm)
        self.k = int(k_history)
        self._buf = None  # [B, k*15] float64

    def reset(self):
        """Clear history; next `build()` fills all K slots with its own sample."""
        self._buf = None

    def build(self, est, goal, prev_action=None, wp1=None, wp2=None):
        """-> float32 [B, 150], oldest sample first."""
        s = normalize(raw_sample(est, goal, prev_action, wp1, wp2),
                      self.offset, self.scale, self.clip_norm)
        B = s.shape[0]
        if self._buf is None or self._buf.shape[0] != B:
            self._buf = np.tile(s, (1, self.k))
        else:
            self._buf = np.concatenate([self._buf[:, RAW_DIM:],
                                        s.reshape(B, RAW_DIM)], axis=1)
        return self._buf.astype(np.float32)
