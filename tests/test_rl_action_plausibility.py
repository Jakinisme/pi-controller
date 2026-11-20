"""Does the vendored ONNX policy behave like a controller AT ALL? — isolated, no Pi, no mock.

Feeds the real ``models/policy.onnx`` (provenance in ``models/MANIFEST.json``) observation
batches built by the real ``src.rl.obs_np`` builder and asserts the returned action is
*commensurate with the error it is given*:

    goal 50 m dead ahead, zero heading error  -> surge meaningfully positive
    goal 50 m exactly astern (180 deg error)  -> surge NOT full positive
    goal 30 m to starboard, heading correct   -> sway sign consistent with "positive sway = starboard"
    large heading error                       -> |yaw| non-trivial
    zero error AT the goal                    -> small action

History: the graph input is fixed at 150 = K=10 samples. ``ObsBuilder.reset()`` fills all
10 slots with the first sample, so every case is fed one raw sample repeated 10x.
Normalization comes from ``models/policy_meta.json`` (== training ``configs/env.yaml``);
the graph itself carries identity ``mean``/``std`` buffers, so this is not double-normalised.

POLICY UNDER TEST — ``a1_long_A`` (A@100M), swapped in on 2026-11-07. Measured with
``tools/describe_policy.py``; the previous R7 numbers are kept in the next block for
comparison, because the asymmetry between the two is the whole point.

``a_prev = 0`` (episode first tick) — the current policy:

    goal                 surge     sway      yaw
    50 m ahead          +0.8387   -0.9551  +0.6782
    50 m astern         +1.0000   +0.2569  -0.9904
    30 m starboard      +0.9636   +0.3729  +0.6472
    30 m port           -0.9375   -0.9665  -0.9938
    0 m (at goal)       +0.0782   -0.0334  -0.1360
    hdg err  90 deg     +0.1544   -0.9615  +0.7944
    grid n=625: surge range [-0.9998, +1.0000], frac(surge>0.99) = 0.768
    per-channel std = [0.4897 0.6628 0.809 ]

R7 (the previous export, for contrast) saturated surge at +1.0 for EVERY geometry including
being AT the goal, was blind to which side the goal sat on (sway difference 0.003 for a
60 m swing), and had per-channel std ~[0 0 0] apart from ``u``.

WHERE THE CURRENT POLICY STANDS — better than R7 on the two criteria R7 failed, still not
a clean goal-tracker:

  * surges to +0.078 AT the goal (R7: +1.0000) -> it does back off, so
    ``test_zero_error_at_goal_commands_a_stop_or_reverse`` now passes;
  * sway separates starboard (+0.3729) from port (-0.9665) at ``a_prev = 0`` -> the
    lateral channel carries goal information, so ``test_sway_...`` now asserts it DOES;
  * BUT under a realistic mid-episode ``a_prev`` the goal still barely moves surge:
    |astern - ahead| = 0.0001 and at-goal surge = +0.9895. It remains a ``a_prev``-dominated
    bang-bang controller; 76.8 % of the grid still sits above surge 0.99. The regime the
    simulator actually visits mid-episode is the one where it is NOT goal-driven.

Read the two ``a_prev`` blocks together before treating either as "the" behaviour: the
policy is much more responsive on the episode's first tick than mid-episode, and the mid
episode regime is the one a running boat lives in.
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


def test_sway_distinguishes_port_from_starboard(session, meta):
    """The current policy's sway DOES follow the side the goal is on — R7's did not.

    Measured at ``a_prev = 0``: goal 30 m to starboard -> sway ``+0.3729``; 30 m to port ->
    ``-0.9665``. A 1.34 swing, and the sign follows "positive sway = starboard". R7 moved
    0.003 for the same 60 m reversal and was blind to the side (see the module docstring).

    Caveat, measured and deliberate: at ``a_prev = A_PREV_ENV`` the split narrows to 0.4362
    and both sides go negative (starboard -0.5581, port -0.9943) — the sign still separates,
    the magnitude collapses. The assertion below is on the ``a_prev = 0`` row because that is
    the row where the channel is cleanly readable; do not read it as "sway is a good lateral
    controller", only as "sway is no longer constant".
    """
    stbd = policy_act(session, meta, dy=30.0)
    port = policy_act(session, meta, dy=-30.0)
    assert stbd[1] - port[1] > 0.1, (
        f"sway no longer separates port from starboard by a meaningful margin: "
        f"{stbd[1]:+.4f} starboard vs {port[1]:+.4f} port "
        f"(split {abs(stbd[1] - port[1]):.4f}); measured split for this export is 1.3394 "
        f"— if it regressed to R7's 0.003 the lateral channel is dead again")


def test_sway_sign_is_starboard_positive(session, meta):
    """The sign convention the DRL mixer assumes: +sway pushes the hull to starboard.

    Measured ``a_prev = 0``: starboard goal -> ``+0.3729`` (positive), port goal -> ``-0.9665``
    (negative). A sign flip here would drive the boat away from its goal on every lateral
    correction, which the asymmetry above would not catch.
    """
    stbd = policy_act(session, meta, dy=30.0)
    port = policy_act(session, meta, dy=-30.0)
    assert stbd[1] > port[1], (
        f"sway sign inverted: starboard goal gave {stbd[1]:+.4f}, port goal gave "
        f"{port[1]:+.4f}; +sway must mean starboard (DRL mixer, INSPECTION H6)")


def test_zero_error_at_goal_commands_a_stop_or_reverse(session, meta):
    """AT the goal with zero error -> the policy must not keep driving forward.

    Measured with ``a_prev = 0``: ``+0.0782`` surge — it stops. R7 held ``+1.0000`` here and
    failed this test; the swap to A@100M is what turns it green.

    Second assertion pins the honest limit: with a realistic mid-episode ``a_prev`` the same
    at-goal input gives ``+0.9895``. Both numbers are measured, and the gap between them is
    the finding — the policy is goal-sensitive on the episode's first tick and ``a_prev``
    dominated afterwards. Asserting only the first row would hide that.
    """
    a = policy_act(session, meta, dx=0.0, dy=0.0)
    assert a[0] < 0.5, (
        f"at goal with zero error the action was surge={a[0]:+.4f} sway={a[1]:+.4f} "
        f"yaw={a[2]:+.4f}; a controller at its goal must not hold forward thrust")

    a_env = policy_act(session, meta, dx=0.0, dy=0.0, a_prev=A_PREV_ENV)
    assert a_env[0] > 0.5, (
        f"the mid-episode at-goal surge is no longer {a_env[0]:+.4f}; this export was measured "
        f"at +0.9895 under A_PREV_ENV — re-measure with tools/describe_policy.py and update "
        f"this test and the module docstring")


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


def test_graph_matches_the_manifest(session, meta):
    """Guard the provenance the tests above depend on.

    ``models/policy.onnx`` must be whatever ``models/MANIFEST.json`` says it is — the
    manifest is written by ``tools/sync_rl_runtime.py`` and is the only record of which DRL
    export is vendored. Every export in this line is 356448 B (identical 256-128-128 MLP,
    150 -> 3), so size proves nothing and the sha256 is the whole check.

    Current export: ``runs/a1_long_A/policy.onnx`` (A@100M), swapped in 2026-11-07; the R7
    export it replaced is kept in ``models/backup_r7_20261107/``.

    The old version of this test hardcoded R7's hash (``dbb90bb8...``) and named
    ``runs/r7_policy.onnx``, so it failed the moment the model changed. Reading the expected
    hash out of the manifest keeps the guard working across exports.
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
