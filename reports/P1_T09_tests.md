# P1 / T09 — test infrastructure + the four core integration tests

Owner of: `tests/conftest.py`, `tests/_fakes.py`, `tests/test_rl_parity.py`,
`tests/test_rl_pid_default.py`, `tests/test_rl_restart.py`,
`tests/test_rl_import_safe.py`. Nothing else touched.

## What was built

| file | contents |
|---|---|
| `tests/conftest.py` | repo root on `sys.path` + `hardware_mock.install()` **before** any `src.*` import (mirrors `test.py`'s header). Fixtures: `state`/`state_factory`, `nav_factory`, `fake_adapter`, `scripted_navigator`, `fake_gps`, `real_adapter` (real `RLAdapter` vs `models/policy.onnx`, skips with reason when model/onnxruntime missing), `drl_path` (skips when absent), path constants, session `real_model_available`. |
| `tests/_fakes.py` | `PolicyStep`, `FakeAdapter` (scripted actions, injectable `inference_ms`, can raise, `reset()` counts, `available`/`reason`/`policy_dt_s`/`meta`/`safety`), `ScriptedNavigator`, `make_state()`, `make_nav()`, `FakeGPS`. Single source of fakes — no test file re-defines any. |
| `tests/test_rl_parity.py` | vendored `RLController` vs DRL's own copy over 200 random synthetic `(est, goal)` sequences (actions within 1e-5 **and** `status["source"]/["fault"]` equal); `MANIFEST.json` sha256 + bytes vs disk; manifest `checkpoint_steps` == meta `steps`; vendored `estimator.py`/`frames.py` byte-identical to DRL. |
| `tests/test_rl_pid_default.py` | `RL_ENABLED=False` **and** config field absent: mux output `==` `ThrustComputer.compute` **field-by-field exact** (`==`, not approx) across a scripted sequence covering WAYPOINT_NAV, STATION_KEEP, IDLE, MANUAL; plus an IDLE-takes-effect-in-one-tick check. |
| `tests/test_rl_restart.py` | fresh `ASVSystem()` → navigator IDLE, `controller.selected == "pid"`, adapter present; a pre-boot `RemoteCommand(mode=waypoint, controller="ppo", timestamp<boot)` is dropped by `_on_command_event`'s timestamp gate and never dispatched; control test proves an after-boot command *is* dispatched with `controller=="ppo"`; pre-boot waypoint command never starts nav. |
| `tests/test_rl_import_safe.py` | two subprocesses: (a) `onnxruntime`/`torch` blocked via a `sys.meta_path` finder → `RLAdapter.available=False` with a reason, `step()` raises `RuntimeError`, `import src.main` OK, PID tick runs; (b) model file missing → same guarantees. Plus an in-process real-`from_config()` PID tick. |

## Verification actually run

```
$ cd E:/buoy-project/pi-control
$ .venv/Scripts/python.exe -m pytest -q
47 passed in 3.97s
EXIT=0
```

**Collection caveat (pre-existing, not introduced here):** the bare
`pytest -q` run collects **47**, not 64. `test.py` is **not** collected by a
bare run because pytest's default `python_files = test_*.py` glob does not match
`test.py` (no `pytest.ini`/`pyproject.toml`/`setup.cfg` exists in the repo). The
baseline command worked only because it named `test.py` explicitly. With the
glob corrected:

```
$ .venv/Scripts/python.exe -m pytest -q -o python_files="test*.py"
64 passed in 3.99s
```

64 = 50 baseline (`test.py` 17 + `test_ml.py` 33) + 14 new. All green.

Per-file / per-fixture checks:

```
$ .venv/Scripts/python.exe -m pytest tests/ -q                       -> 14 passed
$ .venv/Scripts/python.exe -m pytest tests/test_rl_parity.py -q      -> 3 passed
$ .venv/Scripts/python.exe -m pytest tests/test_rl_pid_default.py -q -> 4 passed
$ .venv/Scripts/python.exe -m pytest tests/test_rl_restart.py -q     -> 4 passed
$ .venv/Scripts/python.exe -m pytest tests/test_rl_import_safe.py -q -> 3 passed
$ DRL_PATH=/nonexistent ... pytest tests/test_rl_parity.py -q        -> 1 passed, 2 skipped
$ .venv/Scripts/python.exe -m pytest test.py test_ml.py -q           -> 50 passed (baseline intact)
```

The DRL-absent run confirms the skip path: `test_action_parity_200_sequences`
and `test_vendored_obs_np_is_rewritten_copy` skip with a clear message when
`drl_path` is not a directory — never fail.

## Finding: bare `pytest -q` does not collect `test.py`

Worth flagging to the orchestrator because the task brief and
`p1_interfaces.md` both state the full-suite command is `pytest -q` and that it
"collects `test.py`, `test_ml.py` and `tests/`". It does not collect `test.py`.
Two ways to fix, neither owned by T09:

1. add a `pytest.ini` with `[pytest] python_files = test*.py` (one line), or
2. rename `test.py` → `test_core.py` (not allowed — `test.py` is frozen).

I did **not** create `pytest.ini` because it is outside my ownership list and
would silently change collection for every other owner's run. Recommend the
orchestrator add it (or accept `-o python_files="test*.py"` as the canonical
full-suite command). No test was weakened to work around this.

## Assumptions

- `ControllerMux.compute` is called here with an explicit `config=` dict for the
  "RL off" cases, so the tests do not depend on the deployment `src.config`
  values (`RL_ENABLED=False`) staying put. `test_rl_import_safe.py` additionally
  exercises the real `RLAdapter.from_config()` against `src.config`.
- `test_rl_import_safe.py` drives subprocesses with `sys.executable` (the venv
  Python running pytest), so the child inherits the same interpreter/deps.
- Sys.path juggling for DRL lives only inside `_load_drl_controller` and is
  popped again on the way out; the DRL copy is imported nowhere else.

## Open problems / notes for other owners

1. **No T06 report.** `reports/P1_T06_mux.md` does not exist in the tree (only
   T04/T05/T08 reports are present), though `src/control/controller_mux.py` and
   the T07 main.py wiring are landed. Tests were written against the file's real
   behaviour + the frozen contract; nothing was blocked, but the mux report is
   missing from `reports/`.
2. **T07 raced in during this task.** `src/main.py` initially had no
   `ControllerMux` wiring; mid-task the T07 edit landed (`self.controller`,
   `controller.compute`). `test_rl_restart.py` was written against the landed
   wiring and passes.
3. Bare-`pytest` collection gap — see the finding above.
