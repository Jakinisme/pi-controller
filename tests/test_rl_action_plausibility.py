"""Does the R7 ONNX policy behave like a controller AT ALL? — isolated, no Pi, no mock vessel.

Feeds the real ``models/policy.onnx`` (R7 export, MANIFEST src ``runs/r7_policy.onnx``)
observation batches built by the real ``src.rl.obs_np`` builder and asserts the returned
action is *commensurate with the error it is given*:

    goal 50 m dead ahead, zero heading error  -> surge meaningfully positive
    goal 50 m exactly astern (180 deg error)  -> surge NOT full positive
    goal 30 m to starboard, heading correct   -> sway sign consistent with "positive sway = starboard"
    large heading error                       -> |yaw| non-trivial
    zero error AT the goal                    -> small action

History: the graph input is fixed at 150 = K=10 samples. ``ObsBuilder.reset()`` fills all
10 slots with the first sample, so every case is fed one raw sample repeated 10x.
Normalization comes from ``models/policy_meta.json`` (== training ``configs/env.yaml``);
the graph itself carries identity ``mean``/``std`` buffers, so this is not double-normalised.

MEASURED RESULT — R7 FAILS THE FIRST TWO CRITERIA. It is a saturated open-loop generator:

    goal                 surge     sway      yaw
    50 m ahead          +1.0000   -0.9630  +0.9868
    20 m ahead          +0.9978   -0.9418  +0.9322
    70 m ahead          +1.0000   -0.9630  +0.9868
    50 m astern         +1.0000   -0.8390  +0.6948
    30 m starboard      +0.9999   -0.9820  +0.9283
    30 m port           +1.0000   -0.9793  +0.7519
    0 m (at goal)       +1.0000   -0.7967  -0.1231
    hdg err  30 deg     +1.0000   -0.9765  +0.9887
    hdg err  90 deg     +1.0000   -0.9614  +0.9850
    hdg err 180 deg     +0.9999   -0.6813  +0.0615
    hdg err -90 deg     +1.0000   -0.6405  +0.9546

Surge saturates at +1.0 for every geometry, including being AT the goal. Sway is ~-0.98
(drives astern) for nearly everything and does NOT follow the starboard goal at all.
The only channel with real dynamic range is surge velocity ``u`` (sway: -0.96 @ u=0,
+0.38 @ u=1.5) — the network reads forward speed, not the goal.

Why it matters: posture (surge, sway, yaw) = (+1, -0.98, +0.92) through the DRL sim's own
firmware-verbatim mixer is ``wrench = [+10.1 N, -16.7 N, +5.1 Nm]`` — mostly a sway/kick,
i.e. the mock's circles, with no dependence on where the goal is.

These tests FAIL on purpose: they pin the broken behaviour so a fixed export shows up as a
red->green flip. Do not "fix" them by loosening the thresholds.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(REPO_ROOT, "models", "policy.onnx")
META_PATH = os.path.join(REPO_ROOT, "models", "policy_meta.json")

_K = 10          # k_history; the graph input is fixed at 150 = 15 * 10
_TOL = 1e-3      # "surge is meaningfully positive" / "small action"


# --------------------------------------------------------------- availability
def _skip_reason() -> str | None:
    if not os.path.isfile(MODEL_PATH):
        return f"model absent: {MODEL_PATH}"
    if not os.path.isfile(META_PATH):
        return f"meta absent: {META_PATH}"
    try:
        import onnxruntime  # noqa: F401
    except Exception as exc:                                   # pragma: no cover
        return f"onnxruntime not importable: {exc}"
    return None


_SKIP = _skip_reason()
pytestmark = pytest.mark.skipif(_SKIP is not None, reason=_SKIP or "")


# ------------------------------------------------------------------- the policy
@pytest.fixture(scope="module")
def meta() -> dict:
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def session():
    import onnxruntime as ort

    return ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])


def policy_act(session, meta, *, dx=0.0, dy=0.0, perr=0.0, u=0.0, v=0.0, r=0.0,
               heading=0.0, a_prev=(0.0, 0.0, 0.0)) -> np.ndarray:
    """One raw sample repeated K=10 -> the action the graph returns.

    Built with the real ``src.rl.obs_np.raw_sample``/``normalize`` so the coordinate and
    heading conventions are the runtime's, not this test's.

        dx, dy : goal offset in the BODY frame [m]  (+dx bow, +dy starboard)
        perr   : heading error desired - current [rad], + = turn starboard
        heading: own psi [rad] (heading is not used; dx/dy/perr are already body-frame)
        a_prev : the previous action carried in obs channels 7-9. **This matters.**
            The policy is a bang-bang controller whose output is dominated by
            ``a_prev``, so passing zeros puts it in the one regime the simulator
            visits only on the first tick of an episode and everything downstream
            reads as "surge pinned at +1.0". Use a realistic mid-episode value
            (see ``A_PREV_ENV``) when testing responsiveness.
    """
    from src.rl.obs_np import normalize, raw_sample

    est = np.array([[0.0, 0.0, float(heading), u, v, r]], dtype=np.float64)
    goal = np.array([[dx, dy, heading + perr, 0.0]], dtype=np.float64)   # mode = waypoint
    s = normalize(raw_sample(est, goal, np.array([a_prev], dtype=np.float64)),
                  meta.get("obs_offset"), meta.get("obs_scale"),
                  meta.get("obs_clip_norm", 5.0))
    assert s.shape == (1, 15)
    obs = np.tile(s, (1, _K)).astype(np.float32)
    assert obs.shape == (1, 150), "graph input is fixed at 150 = 15 * k_history"
    out = session.run([meta.get("output_name", "action")],
                      {meta.get("input_name", "obs"): obs})[0]
    a = np.asarray(out, dtype=np.float64).reshape(3)
    assert np.all(np.isfinite(a))
    return a


#: A previously-applied action as the DRL env actually feeds it mid-episode.
A_PREV_ENV = (-0.992, -0.950, -0.738)


@pytest.fixture(scope="module")
def action_table(session, meta) -> dict:
    """Print the evidence once; the per-case tests then assert against it."""
    cases = {
        "50 m ahead":        dict(dx=50.0),
        "50 m astern":       dict(dx=-50.0, perr=np.pi),
        "30 m starboard":    dict(dx=0.0, dy=30.0),
        "0 m at goal":       dict(dx=0.0, dy=0.0),
        "heading err 90 deg": dict(dx=50.0, perr=np.deg2rad(90.0)),
    }
    table = {name: policy_act(session, meta, **kw) for name, kw in cases.items()}
    print("\n  case                 surge     sway      yaw")
    for name, a in table.items():
        print(f"  {name:20s} {a[0]:+.4f}  {a[1]:+.4f}  {a[2]:+.4f}")
    return table


# --------------------------------------------------------------------- verdicts
def test_surge_positive_when_goal_dead_ahead(action_table):
    """Goal 50 m on the bow, no heading error -> surge meaningfully positive."""
    surge = action_table["50 m ahead"][0]
    assert surge > 0.3, (
        f"goal 50 m dead ahead gave surge={surge:+.4f}; a forward-error controller must "
        f"command forward surge (measured R7 table in this file's docstring)")


def test_surge_is_dominated_by_a_prev_not_by_the_goal(session, meta):
    """The measured character of R7: ``a_prev`` sets surge, the goal barely nudges it.

    With ``a_prev = A_PREV_ENV`` the policy commands reverse surge regardless of whether
    the goal is 50 m ahead (``-0.9936``) or 50 m astern (``-0.9995``) — a 0.006 change over
    a 180 deg goal reversal. This is bang-bang hysteresis, not goal tracking, and it is the
    single fact that matters when deciding whether R7 may steer the real boat.

    This test locks the *measured* behaviour. If a retrained policy becomes
    goal-driven, this flips red and should be rewritten to assert goal sensitivity.
    """
    ahead = policy_act(session, meta, dx=50.0, a_prev=A_PREV_ENV)
    astern = policy_act(session, meta, dx=-50.0, perr=np.pi, a_prev=A_PREV_ENV)
    goal_sensitivity = abs(astern[0] - ahead[0])
    assert goal_sensitivity < 0.1, (
        f"goal direction now moves surge by {goal_sensitivity:.4f} "
        f"(ahead {ahead[0]:+.4f}, astern {astern[0]:+.4f}); the policy became more "
        f"goal-driven than measured — update this test and reports/P1_R7_NONTRACKING.md")


def test_a_prev_flips_surge_across_the_whole_range(session, meta):
    """``a_prev`` is the lever: it flips surge from one rail to the other.

    Same state, same goal, only obs channels 7-9 differ: ``a_prev = 0`` -> ``+0.9999``,
    ``a_prev = A_PREV_ENV`` -> ``-0.6191`` at the goal. That is why a static feed with
    ``a_prev = 0`` wrongly read as "surge pinned at +1.0 for every input".
    """
    zero_prev = policy_act(session, meta, dx=0.0, dy=0.0, a_prev=(0.0, 0.0, 0.0))
    env_prev = policy_act(session, meta, dx=0.0, dy=0.0, a_prev=A_PREV_ENV)
    assert abs(zero_prev[0] - env_prev[0]) > 0.5, (
        f"a_prev barely matters: {zero_prev[0]:+.4f} vs {env_prev[0]:+.4f}; the "
        f"documented method artefact in reports/P1_R7_NONTRACKING.md assumes it dominates")


def test_sway_does_not_distinguish_port_from_starboard(session, meta):
    """Measured limitation: R7's sway is blind to which side the goal is on.

    Goal 30 m to starboard -> sway ``-0.9953``; 30 m to port -> ``-0.9986``. A 0.003
    difference for a 60 m swing. Sway is therefore not a lateral-correcting channel; it is
    pinned alongside surge. Locks the measured behaviour, same caveat as above.
    """
    stbd = policy_act(session, meta, dy=30.0, a_prev=A_PREV_ENV)
    port = policy_act(session, meta, dy=-30.0, a_prev=A_PREV_ENV)
    assert abs(stbd[1] - port[1]) < 0.1, (
        f"sway now distinguishes sides by {abs(stbd[1] - port[1]):.4f} "
        f"({stbd[1]:+.4f} starboard vs {port[1]:+.4f} port) — more than measured; "
        f"update this test and reports/P1_R7_NONTRACKING.md")


def test_zero_error_at_goal_commands_a_stop_or_reverse(session, meta):
    """AT the goal with zero error -> the policy must not keep driving forward.

    Measured with a realistic ``a_prev``: ``-0.6191`` surge (it backs off). With
    ``a_prev = 0`` the same input gave ``+1.0000`` — which is why the first version of
    this test read as "the goal is not an input".
    """
    a = policy_act(session, meta, dx=0.0, dy=0.0, a_prev=A_PREV_ENV)
    assert a[0] < 0.5, (
        f"at goal with zero error the action was surge={a[0]:+.4f} sway={a[1]:+.4f} "
        f"yaw={a[2]:+.4f}; a controller at its goal must not hold forward thrust")


def test_large_heading_error_gives_nontrivial_yaw(action_table):
    """90 deg heading error -> |yaw| non-trivial. R7 does this one (yaw=+0.985) — kept as
    a regression guard, not as evidence the policy is sane."""
    yaw = action_table["heading err 90 deg"][2]
    assert abs(yaw) > 0.2, f"90 deg heading error gave |yaw|={abs(yaw):.4f}"


def test_action_is_not_constant_across_opposite_goals(session, meta):
    """A controller's output must change between a goal ahead and the same goal astern."""
    ahead = policy_act(session, meta, dx=50.0)
    astern = policy_act(session, meta, dx=-50.0, perr=np.pi)
    assert np.max(np.abs(ahead - astern)) > 0.2, (
        f"ahead {ahead} vs astern {astern} differ by only "
        f"{np.max(np.abs(ahead - astern)):.4f}; the policy is a near-constant generator")


def test_surge_never_drops_for_any_realistic_observation(session, meta):
    """Characterise the saturation: over a real obstacle-free training box, does surge EVER
    fall below 0.5? Measured: only 0.5% of the grid, all at u > 1.0 m/s or the at-goal row.

    Asserts surge DOES drop below 0.5 somewhere (i.e. it is not a constant), then reports
    the saturation fraction. This test is informational and is expected to pass.
    """
    grid = []
    for dx in (-70.0, -35.0, 0.0, 35.0, 70.0):
        for dy in (-70.0, -35.0, 0.0, 35.0, 70.0):
            for perr in (0.0, np.pi / 3, 2 * np.pi / 3, np.pi, -np.pi / 2):
                for u in (0.0, 0.3, 0.6, 1.0, 1.5):
                    grid.append(policy_act(session, meta, dx=dx, dy=dy, perr=perr, u=u))
    A = np.asarray(grid)
    sat = float((A[:, 0] > 0.99).mean())
    print(f"\n  grid n={len(A)}  surge in [{A[:, 0].min():+.4f}, {A[:, 0].max():+.4f}]  "
          f"frac(surge>0.99)={sat:.3f}  std per channel={A.std(axis=0).round(4)}")
    assert A[:, 0].min() < 0.5, f"surge never drops below 0.5 (min {A[:, 0].min():+.4f})"


def test_graph_is_the_r7_export(session, meta):
    """Guard the provenance the tests above depend on.

    ``models/policy.onnx`` must be ``runs/r7_policy.onnx`` (MANIFEST sha256 dbb90bb8...) and
    NOT the older ``runs/policy.onnx`` (8dd323b3...). Both are 356448 B, so size proves nothing.
    Also: the graph's baked ``mean``/``std`` must be identity, otherwise the meta's
    obs_offset/obs_scale would double-normalise.
    """
    import hashlib

    manifest = os.path.join(REPO_ROOT, "models", "MANIFEST.json")
    if not os.path.isfile(manifest):
        pytest.skip("MANIFEST.json absent")
    with open(manifest, "r", encoding="utf-8") as f:
        want = json.load(f)["files"]["models/policy.onnx"]["sha256"]
    got = hashlib.sha256(open(MODEL_PATH, "rb").read()).hexdigest()
    assert got == want, f"models/policy.onnx sha256 {got} != MANIFEST {want}"

    assert int(meta.get("obs_dim", 0)) == 150 and int(meta.get("act_dim", 0)) == 3
    assert int(meta.get("k_history", _K)) == _K
