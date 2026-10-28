"""Pi-side adapter around the vendored ONNX policy controller.

    RLAdapter  ->  src.rl.policy.RLController (vendored verbatim from drl/runtime/policy.py)

The adapter adds nothing to the control law. It only:

  * builds the controller from ``src.config`` (``RL_MODEL_PATH`` / ``RL_META_PATH``)
    and pins the CPU session to one intra-op thread (Pi 3B has 4 slow cores shared
    with the sensor/telemetry threads);
  * **fails soft**: a missing ``onnxruntime``, a missing model, a corrupt meta file or
    a session that refuses to load never raises out of the constructor. The adapter is
    then ``available == False`` with ``reason`` set and one warning logged, so
    ``src/main.py`` can construct it unconditionally;
  * times the inference of the LAST step and exposes it as ``last_inference_ms`` (the
    mux compares it against ``RL_MAX_INFERENCE_MS``);
  * wraps the controller's status dict in a frozen :class:`PolicyResult`.

``raw_action`` note: ``RLController.step`` returns only the post-clip/post-slew action and
keeps the pre-slew session output in a local variable, so the adapter cannot read it
without touching vendored code. ``raw_action`` therefore **equals** ``action``. If a
future sync exports the raw output, fill it in :meth:`RLAdapter.step` only.

Threading: ``step()`` is called from the 20 Hz control loop. It takes one
``threading.Lock`` for the duration of a step (measured ~1 ms on a laptop; the lock
only serialises accidental concurrent callers, it is not held by anything else), and it
never sleeps or does file I/O.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

__all__ = ["PolicyResult", "RLAdapter"]


@dataclass(frozen=True)
class PolicyResult:
    """One policy step, already clip+slew limited by ``RLController``."""

    action: Tuple[float, float, float]
    raw_action: Tuple[float, float, float]  # == action, see module docstring
    source: str                              # "policy" | "fallback" | "geofence" | "zero" | "init"
    fault: Optional[str]                     # None | "gps" | "sensor" | "watchdog" | "nan" | ...
    inference_ms: float                      # wall time of THIS call


_ZERO3 = (0.0, 0.0, 0.0)


def _cfg_get(cfg: Any, name: str, default: Any) -> Any:
    """Read NAME from a config module or dict; default when absent or None."""
    if cfg is None:
        try:
            from src import config as cfg  # local import: keeps this module import-safe
        except Exception:               # pragma: no cover - config is always there
            return default
    if isinstance(cfg, dict):
        val = cfg.get(name, default)
    else:
        val = getattr(cfg, name, default)
    return default if val is None else val


class RLAdapter:
    """Fail-soft wrapper: construct it always, check ``available`` before use."""

    def __init__(self, model_path=None, meta_path=None, *, enabled=True, log=None,
                 intra_op_num_threads: int = 1, session=None, safety=None,
                 clock=time.monotonic):
        self.log = log or logging.getLogger(__name__)
        self.enabled = bool(enabled)
        self.available = False
        self.reason: Optional[str] = None
        self.last_inference_ms: float = 0.0
        self.meta: Dict[str, Any] = {}
        self.policy_dt_s: float = 0.1
        self.model_path = None if model_path is None else str(model_path)
        self.meta_path = None if meta_path is None else str(meta_path)
        self._clock = clock
        self._lock = threading.Lock()
        self._ctl: Any = None       # RLController (None -> unavailable)
        self._fallback_status = {"source": "init", "fault": None, "action": _ZERO3}

        if not self.enabled:
            # Explicitly disabled: not a failure, so no warning.
            self.reason = "disabled"
            return
        try:
            self._build(intra_op_num_threads, session, safety)
            self.available = True
        except Exception as exc:  # ImportError, OSError, JSONDecodeError, ort errors...
            self.available = False
            self._ctl = None
            self.reason = f"{type(exc).__name__}: {exc}"[:300]
            self.log.warning("RLAdapter unavailable (%s); policy will not run", self.reason)

    # ------------------------------------------------------------------ build
    def _build(self, intra_op_threads, session, safety):
        if session is None:
            import onnxruntime as ort  # ImportError -> unavailable

            if self.model_path is None:
                raise ValueError("no model_path")
            if not Path(self.model_path).is_file():
                raise FileNotFoundError(f"model not found: {self.model_path}")
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = int(intra_op_threads)
            opts.inter_op_num_threads = 1
            session = ort.InferenceSession(
                self.model_path, sess_options=opts,
                providers=["CPUExecutionProvider"])

        # Meta first (own try, so a bad meta is reported as such by RLController too).
        meta: Dict[str, Any] = {}
        if self.meta_path is not None:
            with open(self.meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

        from src.rl.policy import RLController  # ImportError -> unavailable

        self._ctl = RLController(onnx_path=self.model_path, meta_path=self.meta_path,
                                 session=session, safety=safety, clock=self._clock)
        self.meta = dict(getattr(self._ctl, "meta", None) or meta or {})
        self.policy_dt_s = float(self.meta.get("policy_dt_s", 0.1) or 0.1)

    # ------------------------------------------------------------------- step
    def step(self, est: dict, goal: dict) -> PolicyResult:
        """One policy step. Raises ``RuntimeError`` only when unavailable."""
        if not self.available:
            raise RuntimeError(f"adapter unavailable: {self.reason}")
        with self._lock:
            t0 = time.perf_counter()
            action = self._ctl.step(est, goal)          # already clipped + slew limited
            self.last_inference_ms = (time.perf_counter() - t0) * 1000.0
            status = dict(self._ctl.status)
        a = (float(action[0]), float(action[1]), float(action[2]))
        return PolicyResult(action=a, raw_action=a,
                            source=str(status.get("source", "policy")),
                            fault=status.get("fault") or None,
                            inference_ms=self.last_inference_ms)

    def reset(self) -> None:
        """Clear the K=10 ring buffer, ENU origin, (u, v) and the last action."""
        if self._ctl is None:
            return
        with self._lock:
            self._ctl.reset()
        self.last_inference_ms = 0.0

    # ----------------------------------------------------------------- status
    @property
    def status(self) -> Dict[str, Any]:
        """``RLController.status`` passthrough ({"source","fault","action"})."""
        if self._ctl is None:
            return dict(self._fallback_status)
        return dict(self._ctl.status)

    # ------------------------------------------------------------------- ctor
    @classmethod
    def from_config(cls, cfg=None, **kw) -> "RLAdapter":
        """Build from ``src.config`` RL_* names; tolerant of names not defined yet.

        ``RL_ENABLED`` deliberately does NOT gate the model load: it is the mux's
        permission to *use* the policy, not a reason to keep the session out of
        memory. Keeping the adapter loadable means flipping ``RL_ENABLED`` (or a
        dashboard ``controller="ppo"``) needs no restart, and the mux still falls
        back to PID whenever ``RL_ENABLED`` is False — its own rule 1.
        """
        return cls(model_path=_cfg_get(cfg, "RL_MODEL_PATH", None),
                   meta_path=_cfg_get(cfg, "RL_META_PATH", None),
                   enabled=bool(kw.pop("enabled", True)),
                   log=kw.pop("log", None),
                   safety=kw.pop("safety", None),
                   **kw)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"RLAdapter(available={self.available}, reason={self.reason!r}, "
                f"model={self.model_path!r}, dt={self.policy_dt_s})")
