"""Post-processing for the Binet-equation paper: radial period, periapsis
advance, comparison with the weak-field formula, and the Mercury estimate.
Companion to picturesforarticle.py (same normalisation: GM/J^2 = 1)."""
import numpy as np
from scipy.integrate import solve_ivp

GM_J2 = 1.0
E_ECC = 0.5                      # eccentricity of the starting orbit
U0 = (1.0 + E_ECC) * GM_J2       # periapsis: u(0) = (1+e) GM/J^2 = 1.5
RTOL = ATOL = 1e-10

def rhs(coef):
    """u'' = GM/J^2 - u + coef*u^2, coef = 3GM/(c^2 p) (0 -> Newton)."""
    def f(phi, y):
        return [y[1], GM_J2 - y[0] + coef * y[0] ** 2]
    return f

def periapsis_advance(coef, rtol=RTOL, atol=ATOL):
    """Radial period T_r = distance between successive apoapsis passages,
    found as rising zero crossings of du/dphi (u passes through a minimum).
    Integration interval is the same as in picturesforarticle.py."""
    ev = lambda phi, y: y[1]
    ev.direction = 1
    span = (0.0, 15.0 * np.pi)
    sol = solve_ivp(rhs(coef), span, [U0, 0.0], method="RK45",
                    rtol=rtol, atol=atol, events=ev)
    t = sol.t_events[0]
    T_r = np.diff(t)
    return T_r.mean(), T_r.mean() - 2.0 * np.pi

if __name__ == "__main__":
    print("Radial period and periapsis advance, u(0)=1.5, u'(0)=0")
    print("GM/(c^2 p) | coef 3GM/(c^2 p) | dphi_num | dphi_th=6*pi*GM/(c^2 p) | ratio")
    for eps in (0.02, 0.002, 0.0002, 0.00002):
        coef = 3.0 * eps
        T, d_num = periapsis_advance(coef, rtol=1e-12, atol=1e-12)
        d_th = 6.0 * np.pi * eps          # p = a(1-e^2) = J^2/GM = 1
        print(f"{eps:9.5f} | {coef:9.5f} | {d_num:.6f} | {d_th:.6f} | {d_num/d_th:.4f}")

    eps = 0.02
    coef = 3 * eps
    disc = np.sqrt(1 - 4 * coef * GM_J2)
    print("unstable fixed point u* =", (1 + disc) / (2 * coef))
    print("stable fixed point  u_s =", (1 - disc) / (2 * coef))

    # Mercury: dimensionless parameter and observed-scale advance
    GM_sun, c = 1.32712440018e20, 299792458.0
    a, e, P_days = 5.791e10, 0.2056, 87.969
    p = a * (1 - e ** 2)
    eps_m = GM_sun / (c ** 2 * p)
    d_orbit = 6 * np.pi * eps_m
    per_century = d_orbit * (100 * 365.25 / P_days)
    print(f"Mercury: GM/(c^2 p) = {eps_m:.3e}; dphi/orbit = {d_orbit:.3e} rad "
          f"= {np.degrees(d_orbit)*3600:.4f} arcsec; per century = "
          f"{np.degrees(per_century)*3600:.2f} arcsec")
