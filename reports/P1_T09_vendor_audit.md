# P1_T09 — Vendored DRL runtime audit

Scope: `src/rl/{policy,obs_np,estimator,frames}.py` vs `drl/runtime/policy.py`,
`drl/runtime/obs_np.py`, `drl/common/estimator.py`, `drl/common/frames.py`.
Pi repo `E:/buoy-project/pi-control` @ `feat/ppo-controller` / `36fa5ab`; DRL `E:/buoy-project/drl`.

Question answered: has the vendored copy drifted from DRL in any way that **changes
behaviour**, are the **import rewrites** safe, and is the **manifest** trustworthy.

## Method

Three independent layers, so a single shared-rewrite script cannot hide a difference:

1. **Raw text diff** — `diff` of DRL source vs vendored file, no rewrite applied first.
2. **AST diff** — parse both, strip all `Import`/`ImportFrom` nodes, compare
   `ast.dump()` of the remaining bodies. Proves *only* import nodes differ, and that the
   import nodes differ exactly as the documented rewrite says.
3. **Function-level numeric parity** — import both versions in one process
   (Pi: `src.rl.*`; DRL: `common.*`/`runtime.*` as top-level packages) and compare
   outputs on random + boundary inputs.

Scripts (scratch, not committed):
`C:/Users/AORUS/AppData/Local/hermes/cache/scratch/vendor_audit.py`,
`vendor_audit_negctl.py`, `vendor_audit_static.py`, `sync_probe.py`.

Interpreter: `E:/buoy-project/pi-control/.venv/Scripts/python.exe`, numpy + onnxruntime.

### Negative controls (the parity test was proven able to fail)

| control | result |
|---|---|
| Pi and DRL modules are distinct objects from distinct files | distinct (`src\rl\frames.py` vs `drl\common\frames.py`) |
| `cmp()` detects a **1-ulp** change | yes, `max|d| = 8.88e-16` |
| heading-sign convention flip | caught, `max|d| = 6.276` |
| `min_speed_mps` gate straddle (0.3 vs 0.300001 on sog=0.3000005) | caught, `max|d| = 1.0200` |
| reversed K-block ring order | caught, `max|d| = 3.922` |
| attribute set of each module identical both sides | no names only-in-DRL, none only-in-Pi |

## Parity results

| module | functions compared | random cases | max abs diff | verdict |
|---|---|---|---|---|
| `frames.py` | `EARTH_RADIUS_M`, `__all__`, `wrap_deg`, `wrap_pi`, `heading_error_deg`, `heading_error_rad`, `heading_deg_to_psi`, `psi_to_heading_deg`, `latlon_to_enu`, `enu_to_latlon`, `enu_to_body`, `body_to_enu`, `goal_in_body` (11 functions + 2 constants) | 20 016 per angle case (incl. exact 0/±180/±360/±540, ±π, 2π, 3π, 179.999999999, ±1e-12); 20 000 lat/lon cases, lat0 ∈ [-70,70], ±0.05° offsets | **0.0** (bit-identical) | PASS |
| `estimator.py` | `estimate_body_velocity` over `(min_speed_mps, decay_tau_s)` ∈ {(0.3,1.0),(0.0,1.0),(0.3,0.25),(1.5,0.25)} × `dt_s` ∈ {0, 1e-9, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 100.0} + `tau=0` branch; `gps_valid` ∈ {0,1}; sog drawn 0–6 m/s **plus exact gate values** 0.3, 0.2999999999, 0.3000000001, 0.0; cog/heading at ±180/0/360 | 20 006 per config → 198 054 element-comparisons total | **0.0** | PASS |
| `obs_np.py` | constants `RAW_DIM`/`K_HISTORY`/`OBS_DIM`/`DEFAULT_OFFSET`/`DEFAULT_SCALE`/`DEFAULT_CLIP_NORM`, `raw_sample` (full / `prev_action=None` / `wp1=None` / `wp2=None`), `normalize` (default, custom offset+scale+clip, clip edges ±5, 5.0000001), `ObsBuilder` (`build` for 14 consecutive steps, `reset`, batch-size resize, `k_history=3`) | 512-row batches; 14-step ring sequence; 4 clip-edge rows; 30 constant elements | **0.0**; output dtype `float32` both sides | PASS |
| `policy.py` | pure helpers/constants only (`RLController` not constructed): `DEFAULT_SAFETY`, `_MODES`, `__all__` | 3 dict/list equalities | **0.0** (exact equality) | PASS |

`policy.py` logic is *not* directly numerically compared — it needs an ONNX session, and
the task scoped it to pure helpers. Its body carries no rewritable logic (see AST row
below), and every function it calls (`latlon_to_enu`, `heading_deg_to_psi`,
`estimate_body_velocity`, `ObsBuilder`, `normalize`) is covered above.

### Ring-buffer order (K=10), explicitly checked

- First `build()` after `reset()` fills all 10 slots with its own sample (block 0 == block 9).
- For steps 1..13: block `j` of step *i* equals block `j+1` of step *i-1* — i.e.
  **oldest first, newest last**, 150 = 10×15, identical both sides.
- Newest block equals `normalize(raw_sample(...))` cast to float32 — confirms column
  layout, not just mutual agreement.
- `reset()` after fault, batch-size change, and non-default `k_history` all match.

### Import rewrites

Raw diff (no rewrite pre-applied) — the complete set of differences:

```
policy.py  lines 30-32   from common.estimator import  -> from src.rl.estimator import
                         from common.frames import     -> from src.rl.frames import
                         from runtime.obs_np import    -> from src.rl.obs_np import
obs_np.py  line 21       from common.frames import     -> from src.rl.frames import
estimator.py             (no differences)
frames.py                (no differences)
```

| file | import lines | match documented rewrite | non-import AST identical | leftover `common.`/`runtime.` imports |
|---|---|---|---|---|
| `src/rl/policy.py` | 6 | yes | yes | none |
| `src/rl/obs_np.py` | 2 | yes | yes | none |
| `src/rl/estimator.py` | 0 | yes (trivially) | yes | none |
| `src/rl/frames.py` | 1 (`numpy`) | yes (trivially) | yes | none |

Rewrites are safe because:
- The replaced module names (`common.estimator`, `common.frames`, `runtime.obs_np`) are
  used **only** in `from X import ...` form; the replacement is a whole-prefix swap of
  the module path, importing the same symbols from the same code (proven identical above).
- `estimator.py` imports nothing at all and `frames.py` imports only `numpy`, so those
  two files cannot have a namespace-collision issue.
- Remaining literal `common.`/`runtime.` strings are inside docstrings/comments only
  (`obs_np.py:14`, `estimator.py` header, `frames.py` header) — no functional effect.
- Vendored modules under `src.rl.*` and DRL modules under `common.*`/`runtime.*` have no
  name clash; both were imported successfully in the same process for this audit, which
  also demonstrates `src/rl/__init__.py` correctly imports `src.rl.policy`.

`policy.py` and `obs_np.py` both end up importing `src.rl.frames` — same module object,
so no duplicate-module/state divergence.

## Manifest trustworthiness

`models/MANIFEST.json` content checked against disk:

| manifest field | value | independent check |
|---|---|---|
| `drl_commit` | `608256b224dc5e81dc541175df94256399fba4d4` | == `git -C drl rev-parse HEAD` == `origin/master` ✓ |
| `drl_path` | `drl` | relative sibling of `pi-control` ✓ (sync tool recomputes it) |
| `checkpoint_steps` / `iteration` / `source_ckpt` | 299 958 272 / 4577 / `runs/r7_300m/ckpt/ckpt_last.pt` | copied from `r7_policy_meta.json` |
| 4 × `src/rl/*.py` sha256 + bytes | — | **all 6 entries recomputed and match** |
| 2 × `models/policy_*` sha256 + bytes | — | **match** |

### Upstream drift after the vendored commit

- `git -C E:/buoy-project/drl rev-parse HEAD` = `608256b…` — DRL working tree is *at*
  the vendored commit, so **no commit exists after `608256b` at all** (`608256b..HEAD`
  is empty; `git branch -a` shows only `master` and `origin/HEAD -> origin/master ->
  master`; no tags). The last commits touching the four vendored files are
  `1d0a9fd "drl wave 1 done"` (policy, obs_np, estimator) and `a5c44da "drl wave 0
  done"` (frames), both **ancestors** of `608256b`, so their content is what is vendored.
- The only non-clean thing in DRL is `M tests/test_onnx_parity.py` (uncommitted working
  tree modification) — it is **not** one of the four vendored files and nothing is read
  from it by the sync tool. It does mean "DRL is at 608256b" refers to the code state for
  the vendored modules only, not to the whole tree.

### Is the sync tool's `--check` trustworthy?

Probed by running the **real** `tools/sync_rl_runtime.py --check` against a throwaway copy
of the tree (nothing in `pi-control/` was modified):

| probe | exit | detected |
|---|---|---|
| baseline, unmodified copy | 1 | only `drl_path` (expected — temp dir changes the relative root) |
| one constant mutated (`min_speed_mps=0.3` → `0.4`) | 1 | yes: `src/rl/estimator.py: changed` |
| mutation **and** manifest sha256 re-stamped to bless it | 1 | yes: still `changed` + `stale entry` |
| manifest deleted | 1 | yes: `MANIFEST: missing` |
| restored | 1 | only the `drl_path` artefact |

Conclusions: `--check` derives expected bytes from the DRL source and compares them to
disk, so it is **not** a pure text-equality-to-itself check — a blessed manifest cannot
launder a content edit. Verified on the real repo: `--check` exits **0** / `in sync`, and
`git status --porcelain src/rl models tools/sync_rl_runtime.py` is clean.

Known limits (documented, not defects found in practice):
- `--check` compares its own rewrite of DRL against disk; it does **not** validate that
  the rewrite list is correct. That gap is closed by the AST + numeric comparison above,
  not by the tool.
- `drl_path` is recomputed from where the tool is run, so it reports a spurious diff if
  the repo is copied/moved — cosmetic only.
- `--check` compares `drl_commit` to DRL `HEAD`; if DRL gains commits that do **not**
  touch the four vendored files, `--check` would report manifest drift even though the
  vendored behaviour is unchanged. Correct-by-conservative, not a false pass.
- Manifest cannot detect a change that DRL makes *and* re-vendors (a resync updates both
  commit and hashes). Only the parity test above can judge behaviour across a resync.

## Verdict

**Vendored code can be excluded as a cause**: all four modules are bit-identical to DRL
`608256b` modulo three documented import-line rewrites (AST-proven as the only change),
agree to `0.0` max absolute difference over ~300k compared element/values including
wrap boundaries, the SOG gate and the K=10 ring order, no DRL commit after `608256b`
exists, and the manifest's recorded commit and all six sha256 entries verify against disk.
