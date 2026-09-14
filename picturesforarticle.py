import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

GM_J2 = 1.0  
epsilon = 0.05

def newtonian_deriv(phi, y):
    u, u_prime = y
    return [u_prime, GM_J2 - u]

def relativistic_deriv(phi, y):
    u, u_prime = y
    return [u_prime, GM_J2 - u + epsilon * u**2]


y0 = [1.5, 0.0]
phi_span = (0, 15 * np.pi)
phi_eval = np.linspace(phi_span[0], phi_span[1], 3000)

sol_newt = solve_ivp(newtonian_deriv, phi_span, y0, t_eval=phi_eval, rtol=1e-10, atol=1e-10)
sol_rel = solve_ivp(relativistic_deriv, phi_span, y0, t_eval=phi_eval, rtol=1e-10, atol=1e-10)

r_rel = 1.0 / sol_rel.y[0]


plt.figure(figsize=(6, 6), dpi=300)
ax = plt.subplot(111, projection='polar')


ax.plot(sol_rel.t, r_rel, label='Relativistic (Precessing)', color='black', linestyle='-', linewidth=1.2, zorder=2)


phi_single = np.linspace(0, 2 * np.pi, 200)
sol_newt_single = solve_ivp(newtonian_deriv, (0, 2 * np.pi), y0, t_eval=phi_single, rtol=1e-10, atol=1e-10)
r_newt_single = 1.0 / sol_newt_single.y[0]

ax.plot(sol_newt_single.t, r_newt_single, label='Newtonian (Closed Ellipse)', color='black', linestyle='--', dashes=(4, 3), linewidth=1.8, zorder=3)

ax.set_title("Orbital Precession via Perturbed Binet's Equation", color='black', pad=15, fontsize=11, fontweight='bold')
ax.set_rticks([])
ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), fontsize=9)
plt.grid(True, color='gray', alpha=0.3)
plt.savefig('precession.png', bbox_inches='tight', dpi=300)
plt.close()


plt.figure(figsize=(6, 4.2), dpi=300)
plt.plot(sol_rel.y[0], sol_rel.y[1], label='Relativistic', color='black', linestyle='-', linewidth=1.2, zorder=2)
plt.plot(sol_newt.y[0], sol_newt.y[1], label='Newtonian', color='black', linestyle='--', dashes=(4, 3), linewidth=1.6, zorder=3)
plt.xlabel("u (Inverse Radial Distance)", color='black', fontsize=10)
plt.ylabel("du/dφ (Radial Velocity Component)", color='black', fontsize=10)
plt.title("Phase Space Portrait of the Orbital Harmonic Oscillator", color='black', fontsize=11, fontweight='bold')
plt.legend(fontsize=9)
plt.grid(True, color='gray', alpha=0.3)
plt.savefig('phase_portrait.png', bbox_inches='tight', dpi=300)
plt.close()

print("done")
