
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are
from sympy import symbols, sin, cos, simplify, solve, lambdify
from collections import deque

print("Starting Rotary Inverted Pendulum Simulation...")
print("=" * 60)

# ---------------------------
# 1) Symbolic model definition (Furuta-type rotary pendulum)
# ---------------------------
print("Setting up symbolic equations...")
M, m, L1, L2, g, tau = symbols('M m L1 L2 g tau')
th1, w1, th2, w2 = symbols('th1 w1 th2 w2')
a1, a2 = symbols('a1 a2')  # accelerations

# Equations (Euler-Lagrange derived form)
# Note: th2=0 is upright
eq1 = (M + m)*L1**2 * a1 + m*L1*L2 * a2 * cos(th1 - th2) - m*L1*L2 * (w2**2) * sin(th1 - th2) - tau
eq2 = m*L2**2 * a2 + m*L1*L2 * a1 * cos(th1 - th2) + m*g*L2 * sin(th2)

sol = solve([eq1, eq2], (a1, a2), dict=True)[0]
a1_sym = simplify(sol[a1])
a2_sym = simplify(sol[a2])

# Physical parameters
params_vals = {
    M: 0.095,      # arm mass (kg)
    m: 0.024,      # pendulum mass (kg)
    L1: 0.085,     # arm length (m)
    L2: 0.129,     # pendulum length (m)
    g: 9.81
}

# Create numeric functions for accelerations
a1_func = lambdify((th1, w1, th2, w2, tau), a1_sym.subs(params_vals), 'numpy')
a2_func = lambdify((th1, w1, th2, w2, tau), a2_sym.subs(params_vals), 'numpy')

# Extract numeric parameter values
Mval = float(params_vals[M])
mval = float(params_vals[m])
L1val = float(params_vals[L1])
L2val = float(params_vals[L2])
gval = float(params_vals[g])

print(f"Physical parameters: M={Mval}, m={mval}, L1={L1val}, L2={L2val}")

# ---------------------------
# 2) Nonlinear dynamics wrapper
# ---------------------------
def nonlinear_accels(state, u):
    """Compute accelerations using symbolic dynamics."""
    th1_v, w1_v, th2_v, w2_v = state
    a1_v = float(a1_func(th1_v, w1_v, th2_v, w2_v, u))
    a2_v = float(a2_func(th1_v, w1_v, th2_v, w2_v, u))
    return a1_v, a2_v

# ---------------------------
# 3) Numeric linearization about upright (th2=0)
# ---------------------------
def linearize_numeric(x_eq, u_eq=0.0, eps=1e-6):
    """Numerically linearize the system around equilibrium."""
    n = 4
    A = np.zeros((n, n))
    B = np.zeros((n, 1))
    
    def f(x, u):
        a1_v, a2_v = nonlinear_accels(x, u)
        return np.array([x[1], a1_v, x[3], a2_v], dtype=float)
    
    f0 = f(x_eq, u_eq)
    
    # Compute A matrix (partial derivatives w.r.t. state)
    for i in range(n):
        xp = x_eq.copy()
        xp[i] += eps
        fp = f(xp, u_eq)
        A[:, i] = (fp - f0) / eps
    
    # Compute B matrix (partial derivative w.r.t. input)
    up = u_eq + eps
    fup = f(x_eq, up)
    B[:, 0] = (fup - f0) / eps
    
    return A, B

print("Computing numeric linearization...")
x_eq = np.array([0.0, 0.0, 0.0, 0.0])  # upright: th1=0, th2=0
A_num, B_num = linearize_numeric(x_eq)

# Check controllability
def is_controllable(A, B):
    n = A.shape[0]
    ctrb = B
    for i in range(1, n):
        ctrb = np.hstack((ctrb, np.linalg.matrix_power(A, i) @ B))
    return np.linalg.matrix_rank(ctrb) == n

controllable = is_controllable(A_num, B_num)
print(f"System controllable: {controllable}")
print("A_num:\n", np.round(A_num, 4))
print("B_num:\n", np.round(B_num, 4))

# ---------------------------
# 4) LQR design
# ---------------------------
print("\nDesigning LQR controller...")
# Tune Q,R for desired behaviour
Q = np.diag([10.0, 0.1, 200.0, 1.0])   # Heavy weight on pendulum angle
R = np.array([[0.01]])

K = None
try:
    P = solve_continuous_are(A_num, B_num, Q, R)
    K = np.linalg.inv(R) @ B_num.T @ P
    K = np.asarray(K).reshape((1, 4))
    print("LQR K computed:", np.round(K, 4))
    
    # Check closed-loop stability
    A_cl = A_num - B_num @ K
    eigs = np.linalg.eigvals(A_cl)
    print("Closed-loop eigenvalues:", np.round(eigs, 4))
    
except Exception as e:
    print(f"CARE failed: {e}")
    # Fallback gains
    K = np.array([[-5.0, -0.5, -120.0, -8.0]])
    print("Using fallback K:", K)

ACTUATOR_SIGN = 1.0  # Flip if torque direction is reversed

# ---------------------------
# 5) Energy controller (robust from rest)
# ---------------------------
def energy_controller(state):
    """Energy-based swing-up controller with position bias for startup."""
    th1_v, w1_v, th2_v, w2_v = state
    
    # Pendulum energy (upright = 0 convention)
    Jp_eff = mval * (L2val**2)
    E_kin = 0.5 * Jp_eff * (w2_v**2)
    E_pot = mval * gval * L2val * (1 - np.cos(th2_v))
    E_current = E_kin + E_pot
    E_target = 0.0  # Upright energy
    
    E_err = E_target - E_current
    k_e = 1.4  # Energy gain (tune for faster/slower swing)
    u_unsat = k_e * E_err
    
    # Smooth sign term with position bias for startup from rest
    eps = 0.02
    pos_bias = 0.12 * np.sin(th2_v)
    sign_in = w2_v * np.cos(th2_v) + pos_bias
    sign_term = np.tanh(sign_in / eps)
    
    u = float(u_unsat * sign_term)
    
    # Saturation
    u_max = 3.0
    u = ACTUATOR_SIGN * float(np.clip(u, -u_max, u_max))
    
    return u, E_current

# ---------------------------
# 6) Hybrid controller with smooth blending
# ---------------------------
UPRIGHT_THRESH = 0.12   # radians (~7 deg)
VEL_THRESH = 0.8        # rad/s
BLEND_TIME = 0.25       # seconds to blend
_last_enter = None

def controller(t, state):
    """Hybrid controller: energy swing-up -> LQR with blending."""
    global _last_enter
    
    th1_v, w1_v, th2_v, w2_v = state
    th2_n = np.arctan2(np.sin(th2_v), np.cos(th2_v))
    
    # Check if near upright
    in_upright = (abs(th2_n) < UPRIGHT_THRESH) and (abs(w2_v) < VEL_THRESH)
    
    if in_upright:
        if _last_enter is None:
            _last_enter = t
        
        # LQR control
        x_err = np.array([th1_v, w1_v, th2_n, w2_v])
        u_lqr = float(-K @ x_err)
        u_lqr = float(np.clip(u_lqr, -3.0, 3.0))
        
        # Blend from energy to LQR
        if (t - _last_enter) < BLEND_TIME:
            u_energy, E = energy_controller(state)
            w = (t - _last_enter) / BLEND_TIME
            u = (1.0 - w) * u_energy + w * u_lqr
            mode = 'blend'
        else:
            u = u_lqr
            E = 0.0
            mode = 'lqr'
    else:
        _last_enter = None
        u, E = energy_controller(state)
        mode = 'swing'
    
    return float(u), E, mode


def dynamics(t, x):
    """Complete system dynamics with hybrid control."""
    u, E, mode = controller(t, x)
    a1, a2 = nonlinear_accels(x, u)
    return [x[1], a1, x[3], a2]


print("\n" + "=" * 60)
print("Running simulation...")
print("=" * 60)

t0 = 0.0
tf = 15.0
dt = 0.005
t_eval = np.arange(t0, tf + dt, dt)


x0 = np.array([0.0, 0.0, np.pi - 0.08, 0.0])

sol = solve_ivp(dynamics, [t0, tf], x0, t_eval=t_eval, 
                max_step=dt, rtol=1e-6, atol=1e-8)

time = sol.t
states = sol.y.T  # shape (N, 4)

print(f"Simulation completed: {len(time)} time steps")


u_hist = np.zeros(len(time))
E_hist = np.zeros(len(time))
mode_hist = []

for i, (ti, xi) in enumerate(zip(time, states)):
    u_hist[i], E_hist[i], mode = controller(ti, xi)
    mode_hist.append(mode)

mode_hist = np.array(mode_hist)


print("\nCreating visualization...")
print("HellO wORLD")

# Compute 3D positions for animation
arm_len = L1val
pend_len = L2val

x_arm = arm_len * np.cos(states[:, 0])
y_arm = arm_len * np.sin(states[:, 0])
x_p = x_arm + pend_len * np.sin(states[:, 2]) * np.cos(states[:, 0])
y_p = y_arm + pend_len * np.sin(states[:, 2]) * np.sin(states[:, 0])
z_p = pend_len * np.cos(states[:, 2])

# Create figure
fig = plt.figure(figsize=(14, 10))

# Top: 3D animation view
ax_3d = fig.add_subplot(2, 2, 1, projection='3d')
ax_3d.set_xlim([-0.15, 0.15])
ax_3d.set_ylim([-0.15, 0.15])
ax_3d.set_zlim([0, 0.2])
ax_3d.set_xlabel('X (m)')
ax_3d.set_ylabel('Y (m)')
ax_3d.set_zlabel('Z (m)')
ax_3d.set_title('3D View: Rotary Inverted Pendulum')

# Top-down view (X-Y plane)
ax_top = fig.add_subplot(2, 2, 2)
ax_top.set_xlim([-0.25, 0.25])
ax_top.set_ylim([-0.25, 0.25])
ax_top.set_aspect('equal')
ax_top.set_xlabel('X (m)')
ax_top.set_ylabel('Y (m)')
ax_top.set_title('Top-Down View (X-Y Plane)')
ax_top.grid(True, alpha=0.3)

# Bottom plots
ax_angles = fig.add_subplot(2, 2, 3)
ax_control = fig.add_subplot(2, 2, 4)

# Plot angles
theta2_wrapped = np.mod(states[:, 2] + np.pi, 2*np.pi) - np.pi
#ax_angles.plot(time, np.degrees(states[:, 0]), 'b-', linewidth=1.5, label='Arm θ₁')
ax_angles.plot(time, np.degrees(theta2_wrapped), 'r-', linewidth=1.5, label='Pendulum θ₂')
ax_angles.axhline(y=0, color='g', linestyle='--', alpha=0.5, linewidth=1)
ax_angles.set_xlabel('Time (s)')
ax_angles.set_ylabel('Angle (deg)')
ax_angles.set_title('System Angles')
ax_angles.grid(True, alpha=0.3)
ax_angles.legend()

# Plot control and energy
ax_control.plot(time, u_hist, 'k-', linewidth=1.5, label='Control u (Nm)')
ax_control_twin = ax_control.twinx()
ax_control_twin.plot(time, E_hist, 'g-', linewidth=1.5, label='Energy E (J)')
ax_control.set_xlabel('Time (s)')
ax_control.set_ylabel('Control Torque (Nm)', color='k')
ax_control_twin.set_ylabel('Energy (J)', color='g')
ax_control.set_title('Control Input & Energy')
ax_control.grid(True, alpha=0.3)

# Highlight mode switches
lqr_idxs = np.where(mode_hist == 'lqr')[0]
if lqr_idxs.size > 0:
    t_switch = time[lqr_idxs[0]]
    for ax in [ax_angles, ax_control]:
        ax.axvline(t_switch, color='tab:blue', ls='--', alpha=0.7, linewidth=2)
    print(f"\n✓ Switched to LQR at t = {t_switch:.2f} s")

# Add mode regions as background colors
swing_mask = mode_hist == 'swing'
blend_mask = mode_hist == 'blend'
lqr_mask = mode_hist == 'lqr'

# Sample every N points for animation
N_frames = 300
step = max(1, len(time) // N_frames)
anim_indices = range(0, len(time), step)

print(f"\nAnimating {len(anim_indices)} frames...")

# Draw initial frame in 3D
for idx in anim_indices:
    i = idx
    
    # Clear and redraw 3D view
    ax_3d.cla()
    ax_3d.set_xlim([-0.15, 0.15])
    ax_3d.set_ylim([-0.15, 0.15])
    ax_3d.set_zlim([0, 0.2])
    ax_3d.set_xlabel('X (m)')
    ax_3d.set_ylabel('Y (m)')
    ax_3d.set_zlabel('Z (m)')
    
    # Draw arm
    ax_3d.plot([0, x_arm[i]], [0, y_arm[i]], [0, 0], 
               'o-', color='blue', linewidth=4, markersize=8)
    
    # Draw pendulum
    ax_3d.plot([x_arm[i], x_p[i]], [y_arm[i], y_p[i]], [0, z_p[i]], 
               'o-', color='red', linewidth=3, markersize=6)
    
    # Draw trail
    trail_start = max(0, i - 100)
    ax_3d.plot(x_p[trail_start:i], y_p[trail_start:i], z_p[trail_start:i], 
               '-', color='gray', alpha=0.3, linewidth=1)
    
    # Add pivot
    ax_3d.plot([0], [0], [0], 'ko', markersize=10)
    
    # Title with mode
    mode_color = {'swing': 'orange', 'blend': 'yellow', 'lqr': 'green'}
    ax_3d.set_title(f't = {time[i]:.2f}s | Mode: {mode_hist[i].upper()}', 
                    color=mode_color.get(mode_hist[i], 'black'))
    
    # Rotate view
    ax_3d.view_init(elev=20, azim=45 + time[i] * 5)
    
    # Clear and redraw top-down view
    ax_top.cla()
    ax_top.set_xlim([-0.25, 0.25])
    ax_top.set_ylim([-0.25, 0.25])
    ax_top.set_aspect('equal')
    ax_top.set_xlabel('X (m)')
    ax_top.set_ylabel('Y (m)')
    ax_top.set_title('Top-Down View (X-Y Plane)')
    ax_top.grid(True, alpha=0.3)
    
    # Draw in top-down view
    ax_top.plot([0, x_arm[i]], [0, y_arm[i]], 'o-', color='blue', 
                linewidth=6, markersize=8, solid_capstyle='round')
    ax_top.plot([x_arm[i], x_p[i]], [y_arm[i], y_p[i]], 'o-', color='red', 
                linewidth=4, markersize=6, solid_capstyle='round')
    ax_top.plot(x_p[trail_start:i], y_p[trail_start:i], '-', 
                color='gray', alpha=0.4)
    ax_top.plot(0, 0, 'ko', markersize=10)
    
    plt.tight_layout()
    plt.pause(0.001)

print("\n" + "=" * 60)
print("Simulation complete!")
print("=" * 60)

# Final statistics
final_theta2 = theta2_wrapped[-1]
print(f"\nFinal Statistics:")
print(f"  Final arm angle: {np.degrees(states[-1, 0]):.2f}°")
print(f"  Final pendulum angle: {np.degrees(final_theta2):.2f}°")
print(f"  Final arm velocity: {states[-1, 1]:.4f} rad/s")
print(f"  Final pendulum velocity: {states[-1, 3]:.4f} rad/s")

if abs(final_theta2) < np.radians(5):
    print("\n✓ SUCCESS: Pendulum balanced at upright position!")
else:
    print("\n⚠ Pendulum not fully balanced - consider tuning gains")

plt.show()