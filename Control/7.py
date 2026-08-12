import numpy as np
import matplotlib.pyplot as plt

# --- System and Controller Definitions ---
A = np.array([[1.2, 0.3],
              [-0.4, 0.9]])
B = np.array([[0.2],
              [0.8]])
K = np.array([[1.625, 0.96875]]) # From part (a)

x0 = np.array([[10],
               [-5]])
u_limit = 5.0
N_sim = 25 # Simulation length

# --- Simulation Loop ---
# Initialize states
x_lin = x0.copy() # Linear (ideal)
x_sat = x0.copy() # Saturated (actual)

# History lists
u_lin_hist = []
u_sat_hist = []
x1_lin_hist = []
x1_sat_hist = []

for k in range(N_sim):
    # --- Linear (Ideal) Simulation ---
    u_lin = -K @ x_lin
    u_lin_hist.append(u_lin[0,0])
    x1_lin_hist.append(x_lin[0,0])
    x_lin = A @ x_lin + B @ u_lin

    # --- Saturated (Actual) Simulation ---
    u_des = -K @ x_sat # Desired control
    u_act = np.clip(u_des, -u_limit, u_limit) # Apply saturation
    u_sat_hist.append(u_act[0,0])
    x1_sat_hist.append(x_sat[0,0])
    x_sat = A @ x_sat + B @ u_act

# --- Plotting ---
t = np.arange(N_sim)
plt.figure(figsize=(12, 6))

# Plot 1: Control Input u(k)
plt.subplot(1, 2, 1)
plt.plot(t, u_lin_hist, 'r--', label='Ideal $u(k)$ (Linear)')
plt.step(t, u_sat_hist, 'b-', where='post', label='Actual $u(k)$ (Saturated)')
plt.axhline(u_limit, color='k', linestyle=':', label='Limit = 5.0')
plt.axhline(-u_limit, color='k', linestyle=':')
plt.title('Control Input $u(k)$')
plt.xlabel('Time (k)')
plt.ylabel('Control Input')
plt.legend()
plt.grid(True)

# Plot 2: State x1(k)
plt.subplot(1, 2, 2)
plt.plot(t, x1_lin_hist, 'r--', label='Ideal $x_1(k)$ (Linear)')
plt.step(t, x1_sat_hist, 'b-', where='post', label='Actual $x_1(k)$ (Saturated)')
plt.title('State Response $x_1(k)$')
plt.xlabel('Time (k)')
plt.ylabel('State $x_1$')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()