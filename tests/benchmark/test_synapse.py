import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm

from ..utils.file import save_fig


# --- Parameters ---
tau = 2.0  # Time constant
dt = 1.0  # Time step. Try changing this to a larger value (e.g., 0.1)
# to see the increased error in the Forward Euler method.
t_end = 10.0
t_span = np.arange(0, t_end + dt, dt)

# --- Define the ODE System (dy/dt = A * y) ---
# The system of ODEs in matrix form.
# dy/dt = A * y, where y = [y1, y2]^T
# A = [[-1/tau, 1/tau], [0, -1/tau]]
A = np.array([[-1 / tau, 1 / tau], [0, -1 / tau]])


def ode_system(t, y):
    """Defines the right-hand side of the ODE system.

    This function is used by the Forward Euler method.
    """
    return A @ y


# --- Numerical Methods ---


def forward_euler(ode_func, y0, t_span, dt):
    """Implements the Forward Euler method for numerical integration.

    y_{n+1} = y_n + dt * f(t_n, y_n)
    """
    num_steps = len(t_span)
    y_history = np.zeros((2, num_steps))
    y_history[:, 0] = y0

    current_y = y0
    for i in range(num_steps - 1):
        t_current = t_span[i]
        dy_dt = ode_func(t_current, current_y)
        current_y = current_y + dt * dy_dt
        y_history[:, i + 1] = current_y

    return y_history


def exponential_euler(y0, t_span, dt, A):
    """Implements the Exponential Euler method for a linear homogeneous system.
    This method is also known as the matrix exponential method and is exact.

    for linear systems. y_{n+1} = expm(A * dt) * y_n
    """
    num_steps = len(t_span)
    y_history = np.zeros((2, num_steps))
    y_history[:, 0] = y0

    # Pre-compute the matrix exponential term for efficiency.
    exp_term = expm(A * dt)

    current_y = y0
    for i in range(num_steps - 1):
        current_y = exp_term @ current_y
        y_history[:, i + 1] = current_y

    return y_history


# --- Initial Conditions ---
# The delta function input at t=0 causes an instantaneous jump in y2.
# We model this by setting the initial conditions at t=0+.
initial_conditions = np.array([0.0, np.e])

# --- Run the Simulations ---
y_euler = forward_euler(ode_system, initial_conditions, t_span, dt)
y_exp_euler = exponential_euler(initial_conditions, t_span, dt, A)

# --- Exact Analytical Solutions ---
# These are the analytical solutions derived from the ODE system:
# y1(t) = e*(t/tau) * exp(-t/tau)
# y2(t) = e*exp(-t/tau)
y1_exact = np.e * (t_span / tau) * np.exp(-t_span / tau)
y2_exact = np.e * np.exp(-t_span / tau)

# --- Plotting the Results ---
plt.style.use("seaborn-v0_8-whitegrid")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
fig.suptitle("Comparison of Numerical and Exact Solutions", fontsize=16)

# Plot for y1(t)
ax1.plot(
    t_span, y1_exact, label="Exact Solution", linestyle="--", color="k", linewidth=2
)
ax1.plot(t_span, y_euler[0], label="Forward Euler", linestyle="-", color="r", alpha=0.7)
ax1.plot(
    t_span,
    y_exp_euler[0],
    label="Exponential Euler",
    linestyle=":",
    color="b",
    linewidth=3,
)
ax1.set_ylabel("y1(t)", fontsize=12)
ax1.set_title("Solution for y1", fontsize=14)
ax1.legend(loc="best")

# Plot for y2(t)
ax2.plot(
    t_span, y2_exact, label="Exact Solution", linestyle="--", color="k", linewidth=2
)
ax2.plot(t_span, y_euler[1], label="Forward Euler", linestyle="-", color="r", alpha=0.7)
ax2.plot(
    t_span,
    y_exp_euler[1],
    label="Exponential Euler",
    linestyle=":",
    color="b",
    linewidth=3,
)
ax2.set_xlabel("Time (t)", fontsize=12)
ax2.set_ylabel("y2(t)", fontsize=12)
ax2.set_title("Solution for y2", fontsize=14)
ax2.legend(loc="best")

plt.tight_layout(rect=[0, 0, 1, 0.96])
save_fig(fig, "alpha_model_spike")
