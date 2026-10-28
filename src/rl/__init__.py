"""Pi-side RL runtime: vendored DRL policy (numpy + onnxruntime) and its glue.

`RLController` is imported eagerly because `src/rl/policy.py` only imports
onnxruntime *inside* `RLController.__init__`, so `import src.rl` never needs it.
`RLAdapter` and `RLLogger` are resolved lazily (PEP 562) so that importing this
package cannot fail when their dependencies or owners' modules are absent.
"""

from src.rl.policy import RLController

__all__ = ["RLController", "RLAdapter", "RLLogger"]

_LAZY = {"RLAdapter": "src.rl.adapter", "RLLogger": "src.rl.logger"}


def __getattr__(name):
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(mod), name)
