"""Body-velocity estimator: ONE implementation for sim and Pi runtime.

CONTRACT.md §4.10. The real vessel has no (u, v) sensor (INSPECTION.md H7): the only
velocity information is GPS SOG/COG, projected onto the fused heading. This function is
used by `sim/sensors.py` (torch tensors) and must be importable by `runtime/` (numpy
arrays), so it uses *arithmetic operators only* — no `np.`/`torch.` calls, no
`isinstance` branch, no imports. Importing this module pulls in nothing heavy.

Semantics (CONTRACT §4.10):
    dg = wrap180(cog_deg - heading_deg)
    u_new = sog * cos(dg),  v_new = sog * sin(dg)
used only where `gps_valid and sog >= min_speed_mps`; elsewhere the previous estimate
decays first-order toward 0:  x + (0 - x) * min(dt/decay_tau, 1).
A lost fix therefore decays the estimate instead of snapping it to zero.

cos/sin are power series (Chebyshev fit on [-pi, pi], max error 2.1e-11 rad) because a
backend-specific trig call would break the numpy/torch single code path. Only the odd
terms of sin and the even terms of cos are present.
"""

__all__ = ["estimate_body_velocity"]

_DEG2RAD = 0.017453292519943295
_TWO_PI = 6.283185307179586

# sin(x) = x * (S1 + z*(S3 + z*(S5 + ...))),  z = x*x
_S1 = 0.9999999999931206
_S3 = -0.16666666657741108
_S5 = 0.008333333143191407
_S7 = -0.00019841254393578194
_S9 = 2.7556701977011984e-06
_S11 = -2.5038681958844358e-08
_S13 = 1.5896475156346183e-10

# cos(x) = C0 + z*(C2 + z*(C4 + ...)),  z = x*x
_C0 = 1.0000000000000016
_C2 = -0.49999999999675426
_C4 = 0.041666666650821194
_C6 = -0.0013888888663568257
_C8 = 2.4801572909568848e-05
_C10 = -2.7556831146607717e-07
_C12 = 2.0867346465192514e-09
_C14 = -1.1366947817659914e-11


def _sin(x):
    z = x * x
    p = _S13
    p = _S11 + z * p
    p = _S9 + z * p
    p = _S7 + z * p
    p = _S5 + z * p
    p = _S3 + z * p
    p = _S1 + z * p
    return x * p


def _cos(x):
    z = x * x
    p = _C14
    p = _C12 + z * p
    p = _C10 + z * p
    p = _C8 + z * p
    p = _C6 + z * p
    p = _C4 + z * p
    p = _C2 + z * p
    return _C0 + z * p


def _wrap180(a):
    """Wrap degrees to [-180, 180). `%` follows the divisor's sign in numpy and torch."""
    return (a + 180.0) % 360.0 - 180.0


def estimate_body_velocity(sog_mps, cog_deg, heading_deg, dt_s, u_prev, v_prev,
                           gps_valid, min_speed_mps=0.3, decay_tau_s=1.0):
    """GPS SOG/COG + fused heading -> body velocity (u, v) [m/s].

    Args:
        sog_mps: speed over ground [m/s], any shape.
        cog_deg: course over ground [deg], 0 = north, clockwise.
        heading_deg: fused heading [deg], same convention.
        dt_s: estimator update period [s] (float).
        u_prev, v_prev: previous estimate [m/s], same shape as sog_mps.
        gps_valid: bool/0-1 array or scalar, same shape.
        min_speed_mps: below this SOG the COG is meaningless, decay instead.
        decay_tau_s: decay time constant when no usable fix [s].

    Returns:
        (u, v), same shape/broadcast as the inputs. Works for numpy arrays and torch
        tensors, float32 or float64.
    """
    dg = _wrap180(cog_deg - heading_deg) * _DEG2RAD
    sn = _sin(dg)
    cs = _cos(dg)

    # 0/1 gate written with arithmetic only: bool * 1.0 works in numpy and torch.
    use = (gps_valid * 1.0) * ((sog_mps >= min_speed_mps) * 1.0)
    alpha = min(float(dt_s) / float(decay_tau_s), 1.0) if decay_tau_s > 0.0 else 1.0

    u = use * (sog_mps * cs) + (1.0 - use) * (u_prev + (0.0 - u_prev) * alpha)
    v = use * (sog_mps * sn) + (1.0 - use) * (v_prev + (0.0 - v_prev) * alpha)
    return u, v
