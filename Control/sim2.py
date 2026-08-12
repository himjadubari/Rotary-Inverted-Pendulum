import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import RK45
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import control
from collections import deque

# --- 1. Define System Parameters ---
m_arm = 0.095       # Mass of arm (kg)
m_pend = 0.024      # Mass of pendulum (kg)
L_arm = 0.085       # Length of arm to pivot (m)
L_pend = 0.129      # Full length of pendulum (m)
l_pend = L_pend/2   # Distance to center of mass of pendulum (m)
J_arm = 0.000095    # Moment of inertia of arm (kg·m²)
J_pend = 0.0001     # Moment of inertia of pendulum about its center (kg·m²)
g = 9.81            # Gravity (m/s²)
b_arm = 1e-3        # Damping coefficient for arm
b_pend = 1e-3       # Damping coefficient for pendulum

# --- 2. Linearized System Matrices (A, B) ---
# Corrected linearization around theta = 0 (upright position)
m_total = m_arm + m_pend
J_total = J_arm + m_pend * L_arm**2
denom = J_total * (J_pend + m_pend * l_pend**2) - (m_pend * L_arm * l_pend)**2
a23 = -(m_pend**2 * L_arm * l_pend**2 * g) / denom
a43 = (m_pend * l_pend * g * J_total) / denom
b2 = (J_pend + m_pend * l_pend**2) / denom
b4 = -(m_pend * L_arm * l_pend) / denom

A = np.array([
    [0, 1, 0, 0],
    [0, 0, a23, 0],
    [0, 0, 0, 1],
    [0, 0, a43, 0]
])
B = np.array([
    [0],
    [b2],
    [0],
    [b4]
])

# --- 3. LQR Design (UPDATED) ---
# We don't care about absolute arm angle 'alpha', only 'alpha_dot'
# Penalize: [alpha, alpha_dot, theta, theta_dot]
Q = np.diag([0.0, 1.0, 10.0, 0.1])  # <-- alpha penalty is 0.0, alpha_dot is 1.0
R = np.array([[1.0]])               # Increased control effort penalty

# Calculate LQR gain
K, S, E = control.lqr(A, B, Q, R)
print("LQR Gain K:", K)
print("Closed-loop eigenvalues:", E)

# --- 4. Non-Linear Equations of Motion ---
def non_linear_equations(state, u):
    alpha, alpha_dot, theta, theta_dot = state
    M11 = J_arm + m_pend * L_arm**2
    M12 = m_pend * L_arm * l_pend * np.cos(theta)
    M21 = M12
    M22 = J_pend + m_pend * l_pend**2
    M = np.array([[M11, M12], [M21, M22]])
    C1 = -m_pend * L_arm * l_pend * theta_dot**2 * np.sin(theta) - b_arm * alpha_dot + u
    C2 = m_pend * l_pend * g * np.sin(theta) - b_pend * theta_dot
    C = np.array([C1, C2])
    try:
        accelerations = np.linalg.solve(M, C)
    except np.linalg.LinAlgError:
        print("Singular matrix encountered, returning zero acceleration.")
        accelerations = np.zeros(2)
    return accelerations[0], accelerations[1]

# --- 5. Improved Energy-Based Swing-Up Controller ---
def energy_controller(state):
    """
    Energy-based swing-up controller with smooth control law.
    Using E_potential = m*g*l*(1 - cos(theta)) so theta=0 => E=0
    """
    alpha, alpha_dot, theta, theta_dot = state
    
    # Target energy (zero at upright position theta=0)
    E_target = 0.0
    
    # Current energy
    E_kinetic = 0.5 * (J_pend + m_pend * l_pend**2) * theta_dot**2
    E_potential = m_pend * g * l_pend * (1 - np.cos(theta))
    E_current = E_kinetic + E_potential
    
    # Energy error
    E_error = E_target - E_current
    
    # Gentler gain and smooth sign function to avoid chattering
    k_energy = 1.0
    u_unsat = k_energy * E_error
    
    # Smooth sign function using tanh
    eps = 0.1
    sign_term = np.tanh((theta_dot * np.cos(theta)) / eps)
    u = u_unsat * sign_term
    
    # Clip to actuator limits
    u_max = 2.0
    u = float(np.clip(u, -u_max, u_max))
    
    return u

# --- 6. Hybrid Controller with Improved Switching (UPDATED) ---
# Tightened thresholds for better performance
UPRIGHT_THRESHOLD = 0.2   # rad (~11.5 degrees)
VELOCITY_THRESHOLD = 1.0  # rad/s

# Blending parameters for smooth transition
BLEND_TIME = 0.2  # seconds
blend_start_time = None
blend_u_energy = 0.0

def system_dynamics(t, state):
    """
    Hybrid controller: Energy swing-up + LQR balancing with smooth blending.
    """
    global blend_start_time, blend_u_energy
    
    alpha, alpha_dot, theta, theta_dot = state
    
    # Normalize theta to [-pi, pi]
    theta_normalized = np.arctan2(np.sin(theta), np.cos(theta))
    
    # Check if pendulum is near upright and slow enough
    near_upright = abs(theta_normalized) < UPRIGHT_THRESHOLD and abs(theta_dot) < VELOCITY_THRESHOLD
    
    if near_upright:
        # MODE 1: LQR Balancing (with optional blending)
        
        # --- UPDATE ---
        # We don't care about absolute alpha, so its error contribution is 0
        state_error = np.array([0.0, alpha_dot, theta_normalized, theta_dot])
        
        u_lqr = -K @ state_error
        u_lqr = float(u_lqr[0])
        u_lqr = float(np.clip(u_lqr, -2.0, 2.0))
        
        # Smooth blending from energy to LQR
        if blend_start_time is None:
            # Just entered LQR zone, start blending
            blend_start_time = t
            blend_u_energy = energy_controller(state)
        
        # Calculate blend weight
        time_in_blend = t - blend_start_time
        if time_in_blend < BLEND_TIME:
            w = time_in_blend / BLEND_TIME  # Ramp from 0 to 1
            u = (1 - w) * blend_u_energy + w * u_lqr
        else:
            u = u_lqr  # Pure LQR after blend period
    else:
        # MODE 2: Energy Swing-Up
        blend_start_time = None  # Reset blending when leaving LQR zone
        u = energy_controller(state)
    
    # Compute accelerations using non-linear dynamics
    alpha_ddot, theta_ddot = non_linear_equations(state, u)
    
    return [alpha_dot, alpha_ddot, theta_dot, theta_ddot]

# --- 7. Live Simulation Generator (UPDATED) ---
def simulation_generator():
    """
    Generator that yields the simulation state at each step.
    """
    dt = 0.02  # 50 FPS
    
    # Initial state: pendulum hanging down with small perturbation
    initial_state = [0.0, 0.0, np.pi - 0.1, 0.0]
    
    # Initialize the RK45 solver
    solver = RK45(system_dynamics, 
                  t0=0.0, 
                  y0=initial_state, 
                  t_bound=np.inf,
                  max_step=dt)
    
    while True:
        try:
            solver.step()
            
            # --- CRITICAL UPDATE ---
            # "Wrap" the pendulum angle after each step to keep it in [-pi, pi]
            # This prevents a mismatch between the controller and the physics
            solver.y[2] = np.arctan2(np.sin(solver.y[2]), np.cos(solver.y[2]))
            # ---------------------
            
            yield solver.t, solver.y
            
        except Exception as e:
            print(f"Error in solver: {e}")
            break

# --- 8. Live 3D Animation Setup ---
# (No changes to this section)

# Set up the figure
fig_anim = plt.figure(figsize=(14, 10))

# Create 3D subplot
ax_3d = fig_anim.add_subplot(121, projection='3d')
ax_3d.set_xlim([-0.2, 0.2])
ax_3d.set_ylim([-0.2, 0.2])
ax_3d.set_zlim([0, 0.25])
ax_3d.set_xlabel('X (m)', fontsize=10)
ax_3d.set_ylabel('Y (m)', fontsize=10)
ax_3d.set_zlabel('Z (m)', fontsize=10)
ax_3d.set_title('LIVE 3D Rotary Pendulum (Swing-Up + LQR)', fontsize=12, fontweight='bold')

# Create 2D state plots
ax_states = fig_anim.add_subplot(322)
ax_theta = fig_anim.add_subplot(324)
ax_velocities = fig_anim.add_subplot(326)

# Initialize plot lines
line_alpha, = ax_states.plot([], [], 'b-', linewidth=2, label='Arm α')
line_theta, = ax_theta.plot([], [], 'r-', linewidth=2, label='Pendulum θ')
line_alpha_dot, = ax_velocities.plot([], [], 'b-', linewidth=2, label='α_dot')
line_theta_dot, = ax_velocities.plot([], [], 'r-', linewidth=2, label='θ_dot')

# Setup 2D plot properties
for ax in [ax_states, ax_theta, ax_velocities]:
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

ax_states.set_ylabel('Arm Angle α (deg)', fontsize=9)
ax_theta.set_ylabel('Pendulum θ (deg)', fontsize=9)
ax_velocities.set_xlabel('Time (s)', fontsize=9)
ax_velocities.set_ylabel('Angular Velocity (rad/s)', fontsize=9)

# Threshold indicators
ax_theta.axhline(y=0, color='g', linestyle='--', alpha=0.5, linewidth=1, label='Upright')
ax_theta.axhline(y=11.5, color='orange', linestyle='--', alpha=0.5, linewidth=1, label='LQR Threshold')
ax_theta.axhline(y=-11.5, color='orange', linestyle='--', alpha=0.5, linewidth=1)

# 3D visualization elements
arm_line, = ax_3d.plot([], [], [], 'o-', c='blue', lw=4, markersize=8, label='Arm')
pendulum_line, = ax_3d.plot([], [], [], 'o-', c='red', lw=3, markersize=6, label='Pendulum')
pivot_point, = ax_3d.plot([0], [0], [0], 'ko', markersize=12)
trail_line, = ax_3d.plot([], [], [], 'r-', alpha=0.3, linewidth=1)

time_text = ax_3d.text2D(0.05, 0.95, '', transform=ax_3d.transAxes, fontsize=12, 
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Data history using deques for efficiency
plot_window_s = 10.0  # Show last 10 seconds
max_history_len = int(plot_window_s / 0.02)

t_history = deque(maxlen=max_history_len)
alpha_history = deque(maxlen=max_history_len)
theta_history = deque(maxlen=max_history_len)
alpha_dot_history = deque(maxlen=max_history_len)
theta_dot_history = deque(maxlen=max_history_len)
trail_x = deque(maxlen=100)
trail_y = deque(maxlen=100)
trail_z = deque(maxlen=100)

def init():
    """Initialize animation."""
    arm_line.set_data([], [])
    arm_line.set_3d_properties([])
    pendulum_line.set_data([], [])
    pendulum_line.set_3d_properties([])
    trail_line.set_data([], [])
    trail_line.set_3d_properties([])
    line_alpha.set_data([], [])
    line_theta.set_data([], [])
    line_alpha_dot.set_data([], [])
    line_theta_dot.set_data([], [])
    time_text.set_text('')
    
    return (arm_line, pendulum_line, trail_line, line_alpha, line_theta, 
            line_alpha_dot, line_theta_dot, time_text)

def animate(data):
    """Animation update function."""
    current_time, state = data
    alpha, alpha_dot, theta, theta_dot = state
    
    # --- 3D Coordinates ---
    x_arm = L_arm * np.cos(alpha)
    y_arm = L_arm * np.sin(alpha)
    z_arm = 0.0
    
    x_pend = x_arm + l_pend * np.sin(theta) * np.cos(alpha)
    y_pend = y_arm + l_pend * np.sin(theta) * np.sin(alpha)
    z_pend = l_pend * np.cos(theta)
    
    # Update 3D lines
    arm_line.set_data([0, x_arm], [0, y_arm])
    arm_line.set_3d_properties([0, z_arm])
    pendulum_line.set_data([x_arm, x_pend], [y_arm, y_pend])
    pendulum_line.set_3d_properties([z_arm, z_pend])
    
    # Update trail
    trail_x.append(x_pend)
    trail_y.append(y_pend)
    trail_z.append(z_pend)
    trail_line.set_data(list(trail_x), list(trail_y))
    trail_line.set_3d_properties(list(trail_z))
    
    # --- Update 2D Plot History ---
    theta_normalized = np.arctan2(np.sin(theta), np.cos(theta))
    
    t_history.append(current_time)
    alpha_history.append(np.degrees(alpha))
    theta_history.append(np.degrees(theta_normalized))
    alpha_dot_history.append(alpha_dot)
    theta_dot_history.append(theta_dot)
    
    # Update 2D state plots
    line_alpha.set_data(t_history, alpha_history)
    line_theta.set_data(t_history, theta_history)
    line_alpha_dot.set_data(t_history, alpha_dot_history)
    line_theta_dot.set_data(t_history, theta_dot_history)
    
    # Rescale 2D plots
    if len(t_history) > 1:
        t_min = t_history[0]
        t_max = t_history[-1]
        
        for ax in [ax_states, ax_theta, ax_velocities]:
            ax.set_xlim(t_min, t_max)
        
        # Safe min/max with fallbacks
        alpha_min = min(alpha_history) if alpha_history else 0
        alpha_max = max(alpha_history) if alpha_history else 0
        theta_min = min(theta_history) if theta_history else -180
        theta_max = max(theta_history) if theta_history else 180
        
        ax_states.set_ylim(alpha_min - 5, alpha_max + 5)
        ax_theta.set_ylim(max(theta_min - 10, -200), min(theta_max + 10, 200))
        
        vel_min = min(min(alpha_dot_history) if alpha_dot_history else 0, 
                      min(theta_dot_history) if theta_dot_history else 0)
        vel_max = max(max(alpha_dot_history) if alpha_dot_history else 0,
                      max(theta_dot_history) if theta_dot_history else 0)
        ax_velocities.set_ylim(vel_min - 1, vel_max + 1)
    
    # Update time text
    time_text.set_text(f'Time = {current_time:.2f} s')
    
    # Rotate 3D view
    ax_3d.view_init(elev=20, azim=45 + current_time * 2)
    
    return (arm_line, pendulum_line, trail_line, line_alpha, line_theta, 
            line_alpha_dot, line_theta_dot, time_text)

# --- 9. Run the Live Animation (UPDATED) ---

print("\n" + "="*60)
print("Starting LIVE Rotary Inverted Pendulum Simulation")
print("="*60)
print("\nImprovements implemented:")
print("   ✓ Corrected LQR cost (alpha penalty = 0.0)")
print("   ✓ Corrected LQR error state (alpha error = 0.0)")
print("   ✓ Added angle 'wrapping' to solver loop for stability")
print("   ✓ Smooth energy controller (tanh sign function)")
print("   ✓ Tightened switching thresholds")
print("   ✓ Smooth blending between controllers")
print("   ✓ Added damping for stability")
print("\nClose the plot window to stop the simulation.")
print("="*60 + "\n")

# Create the generator
sim_gen = simulation_generator()

# Create animation
ani = FuncAnimation(fig_anim, 
                    animate, 
                    frames=sim_gen,
                    interval=20,       # 20ms = 50 FPS
                    blit=False,        # Must be False for 3D
                    init_func=init,
                    save_count=0)      # Prevents memory leak

plt.tight_layout()
plt.show()

print("\nSimulation stopped.")