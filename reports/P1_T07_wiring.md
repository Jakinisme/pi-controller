# T07 — ControllerMux wiring in main.py + Firebase `controller` field & telemetry keys

Files owned + changed: `src/main.py` (UTF-8 **CRLF**, 18+/1−), `src/comms/firebase_bridge.py` (UTF-8 **LF**, 19+/0−), this report.

## What changed

### `src/main.py` (`ASVSystem`) — 4 hunks
1. Imports: `from src.control.controller_mux import ControllerMux` + `from src.rl.adapter import RLAdapter` (after the firebase_bridge import).
2. `__init__`: after `self.thrust = ThrustComputer()`,
   `self.rl_adapter = RLAdapter.from_config()` then
   `self.controller = ControllerMux(self.thrust, self.rl_adapter, self.navigator, self.gps)`.
   **No `rl_logger` passed** → the mux keeps `rl_logger=None`, i.e. it does *not* own an RLLogger in this wiring, so **no shutdown change** was needed (`T08`'s `RLLogger` + a background flusher are a separate concern; wiring one here would put disk ownership in the mux and require a shutdown flush we were told not to add unless the mux owns one).
   `self.thrust` kept verbatim — `_handle_remote_command` still uses it for `reset_all()` / `set_manual_thrust()`.
3. `_handle_remote_command`: first statement after the log line is
   `self.controller.set_controller_request(cmd.controller)`. Every existing branch (`idle`/`station_keep`/`waypoint`/`manual`/unknown) is byte-identical.
4. `control_loop`: `thrust_cmd = self.controller.compute(nav_output, state)` replaces the `self.thrust.compute(...)` call; plus one `self.firebase.push_control(self.controller.selected, self.controller.source, self.controller.fault)` right after `push_nav(...)`.

### `src/comms/firebase_bridge.py` — 4 hunks
- Module docstring: documents `/telemetry/control` and the `controller` command field.
- `RemoteCommand.controller: str = "pid"`; `from_dict` does `data.get("controller", "pid")` and normalises anything ∉ `{"pid","ppo"}` to `"pid"` with **one** `log.warning` (`"Unknown controller %r in command -> 'pid'"`).
- New `FirebaseBridge.push_control(controller, source, fault=None)` → `queue_telemetry("telemetry/control", {controller, source, fault})`, following the `push_nav`/`push_anomaly` helper style. **Least invasive node choice:** a *new additive* `telemetry/control` sub-node — the brief offered "a `control` sub-node **or** on the existing `nav` node"; a new sibling node touches no existing node's payload shape (the existing `_send_telemetry` flattens `{path: {k:v}}` → `path/k`, so an extra key on `nav` would have *changed* the `nav` node's schema, whereas a new node changes nothing existing). `fault=None` is serialised as `""` (Firebase RTDB rejects `None`).
- **Untouched on purpose:** `_last_command_ts` and both gate call sites (`_on_command_event`, `_listen_for_commands`) — no line changed; `controller` is parsed **only** inside `from_dict`, never on startup, never outside the timestamp gate.

## Verification (real output)

### 1. Diff size — no whole-file churn
```
$ git diff --stat src/main.py src/comms/firebase_bridge.py
 src/comms/firebase_bridge.py | 19 +++++++++++++++++++
 src/main.py                  | 19 ++++++++++++++++++-
 2 files changed, 37 insertions(+), 1 deletion(-)

$ git diff --numstat src/main.py src/comms/firebase_bridge.py
19      0       src/comms/firebase_bridge.py
18      1       src/main.py
```
(`git` prints `warning: in the working copy of 'src/comms/firebase_bridge.py', LF will be replaced by CRLF the next time Git touches it` — this is the repo's autocrlf normalisation at *commit* time, not a working-tree change. The file on disk is still LF, as required. `git diff` shows 19 pure insertions, i.e. no line-ending churn.)

### 2. Line endings preserved
```
$ file src/main.py src/comms/firebase_bridge.py
src/main.py:                  Python script, Unicode text, UTF-8 text executable, with CRLF line terminators
src/comms/firebase_bridge.py: Python script, Unicode text, UTF-8 text executable
```
main.py still CRLF; firebase_bridge.py still LF.

### 3. Fresh `ASVSystem()` starts IDLE + PID
```
$ .venv/Scripts/python.exe -c "..."   # hardware_mock.install(); from src.main import ASVSystem
{
  "nav_state": "IDLE",
  "selected": "pid",
  "adapter_available": false,
  "adapter_reason": "disabled",
  "has_thrust_attr": true,
  "check1": "PASS navigator=IDLE controller.selected=pid",
  ...
}
```
Notes: `RLAdapter.from_config()` → `available=False, reason='disabled'` (RL_ENABLED=False shipped default). The mux handles this itself: `_resolve_selected` requires `enabled AND wants_ppo AND available`, so `selected="pid"`, and because `RL_ENABLED` is False the `unavailable` latch branch (`enabled and self._request=="ppo" and not available`) does **not** fire — no latch on a disabled deployment. main.py therefore stays simple: one unconditional `RLAdapter.from_config()`, no enable-branch.

### 4. Restart safety — stale pre-boot `waypoint`+`ppo` dropped
```
$ .venv/Scripts/python.exe -c "..."   # FirebaseBridge(enabled=False)
{
  "parsed_mode": "waypoint",
  "parsed_controller": "ppo",
  "gate_dropped": true,
  "check2": "PASS stale waypoint+ppo (ts<boot) dropped by _last_command_ts gate; fresh system IDLE/pid",
  "bogus_norm": "pid",
  "absent_controller": "pid",
  "check3": "PASS unknown->pid, absent->pid",
  "selected_after_ppo_req_rl_disabled": "pid",
  "check4": "PASS ppo request with RL_ENABLED=False stays pid (safety)",
  "check5": "PASS push_control -> telemetry/control {controller,source,fault}"
}
```
The gate logic exercised is the real code path: `float(cmd.timestamp) <= bridge._last_command_ts` (identical comparison to `_on_command_event:292` / `_listen_for_commands:247`), on a `RemoteCommand.from_dict` built exactly as the stream callback builds it. Result: dropped → handler never runs → fresh `ASVSystem()` stays IDLE/PID. `from_dict` itself does **not** drop a `ppo` field (it parses it faithfully) — the *gate* drops the whole stale command, which is the required defence.

### 5. `set_controller_request` takes effect within one tick (mux contract check)
```
$ .venv/Scripts/python.exe ...   # RL_ENABLED=True + available fake adapter
Unknown controller request 'bogus' -> 'pid'
fresh selected = pid
after ppo request, one tick -> selected = ppo source = policy
after unknown request, one tick -> selected = pid
MUZZLE_OK
```
Confirms the main.py wiring point: `set_controller_request` records the request + resets RL state; `selected` is recomputed by `_resolve_selected()` inside the next `compute()` call — so the request is live on the **next control tick**, which satisfies "takes effect within one control tick, including while the policy is active".

### 6. Existing suite unchanged + compiles
```
$ .venv/Scripts/python.exe -m pytest test.py test_ml.py -q
50 passed in 1.87s

$ .venv/Scripts/python.exe -m py_compile src/main.py src/comms/firebase_bridge.py
(exit 0)  PY_COMPILE_OK
```

## Which test locks this behaviour (T09/T10 own `tests/` — not written here)
- **`tests/test_rl_restart.py`** is the one that must keep this locked: it should assert a fresh `ASVSystem()` (or `WaypointNavigator`+`ControllerMux`) reports `navigator.state is NavigationState.IDLE` and `controller.selected == "pid"` with `RL_ENABLED=False`, and that a `RemoteCommand.from_dict({"mode":"waypoint","controller":"ppo","timestamp":<past>})` handed through the bridge's `_last_command_ts` comparison is dropped (no switch to ppo, still IDLE/PID).
- **`tests/test_rl_pid_default.py`** should additionally assert `RemoteCommand.from_dict` normalisation (absent → `"pid"`, unknown → `"pid"` + one warning) and that `push_control` writes `telemetry/control` with `controller`/`source`/`fault`.
- **`tests/test_rl_parity.py`** covers the swapped `compute()` call (mux PID output byte-identical to raw `ThrustComputer.compute` when not selected), which is what makes the `control_loop` change safe.

## Open problems / assumptions
1. **`push_control` is called every control tick (20 Hz)** but the bridge's `queue_telemetry` overwrites the same path key, and `_send_telemetry` is rate-limited to `FIREBASE_LOG_RATE_HZ` — so the node is pushed at the telemetry rate, same as the other `push_*` helpers. No cost added beyond one dict write per tick.
2. **No `rl_logger` is wired** (documented above). If the deployment wants the CSV trace, someone must construct `RLLogger(RL_LOG_DIR)` and pass it as `rl_logger=` to the `ControllerMux(...)` call *and* add a shutdown `flush()/close()` — both are outside this task's "minimal diff" and this task's spec did not mandate them (`§main.py` froze the constructor call without a logger).
3. **fault telemetry `""` not `null`**: Firebase RTDB rejects `None` values, so a `None` fault is serialised as `""` (mirrors the RLLogger CSV convention). A consumer must treat `""` as "no fault".
4. The mux's `selected` attribute reflects the controller **after** the last `compute()`; before the first tick it stays `"pid"` even if `ppo` was requested. Not a bug — `_resolve_selected()` runs in `compute()`. Telemetry therefore reports `pid` until the first tick, which is correct for a fresh boot.
