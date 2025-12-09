import numpy as np
import sympy as smp
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib import animationo
from matplotlib.animation import PillowWriter

# 1. Symbolic Setup
t, g = smp.symbols('t g')
m1, m2 = smp.symbols('m1 m2')
L1, L2 = smp.symbols('L1, L2')

# The two angles are functions of time
the1, the2 = smp.symbols(r'\theta_1, \theta_2', cls=smp.Function)
the1 = the1(t)
the2 = the2(t)

# Derivatives
the1_d = smp.diff(the1, t)
the2_d = smp.diff(the2, t)
the1_dd = smp.diff(the1_d, t)
the2_dd = smp.diff(the2_d, t)

# Cartesian Coordinates (Symbolic)
x1_smp = L1 * smp.sin(the1)
y1_smp = -L1 * smp.cos(the1)
x2_smp = L1 * smp.sin(the1) + L2 * smp.sin(the2)
y2_smp = -L1 * smp.cos(the1) - L2 * smp.cos(the2)

# Set up Lagrangian
T1 = 0.5 * m1 * (smp.diff(x1_smp, t)**2 + smp.diff(y1_smp, t)**2)
T2 = 0.5 * m2 * (smp.diff(x2_smp, t)**2 + smp.diff(y2_smp, t)**2)
T = T1 + T2

V1 = m1 * g * y1_smp
V2 = m2 * g * y2_smp
V = V1 + V2

L = T - V

# Get the Lagrange equations
LE1 = smp.diff(L, the1) - smp.diff(smp.diff(L, the1_d), t).simplify()
LE2 = smp.diff(L, the2) - smp.diff(smp.diff(L, the2_d), t).simplify()

# Solve for second derivatives (accelerations)
sols = smp.solve([LE1, LE2], (the1_dd, the2_dd), simplify=False, rational=False)

# 2. Convert to Numerical Functions (Lambdify)

# Functions for the second derivatives (z1_dot and z2_dot)
dz1dt_f = smp.lambdify((t, g, m1, m2, L1, L2, the1, the2, the1_d, the2_d), sols[the1_dd], 'numpy')
dz2dt_f = smp.lambdify((t, g, m1, m2, L1, L2, the1, the2, the1_d, the2_d), sols[the2_dd], 'numpy')

# Functions for the first derivatives (the1_dot and the2_dot, which are just z1 and z2)
dthe1dt_f = smp.lambdify(the1_d, the1_d, 'numpy')
dthe2dt_f = smp.lambdify(the2_d, the2_d, 'numpy')

# Define S = [the1, z1, the2, z2] where z1=the1_d and z2=the2_d
def dSdt(S, t, g, m1, m2, L1, L2):
    the1_num, z1_num, the2_num, z2_num = S
    return [dthe1dt_f(z1_num),
            dz1dt_f(t, g, m1, m2, L1, L2, the1_num, the2_num, z1_num, z2_num),
            dthe2dt_f(z2_num),
            dz2dt_f(t, g, m1, m2, L1, L2, the1_num, the2_num, z1_num, z2_num),
            ]

# 3. Numerical Integration (odeint)
t_num = np.linspace(0, 40, 1001) # Renamed to t_num to avoid conflict
g_num = 9.8
m1_num = 2
m2_num = 1
L1_num = 2
L2_num = 1
# Initial conditions: [the1(0), z1(0), the2(0), z2(0)] = [1, -3, -1, 5]
ans = odeint(dSdt, y0=[1, -3, -1, 5], t=t_num, args=(g_num, m1_num, m2_num, L1_num, L2_num))

# Extract the numerical angle arrays
the1_arr = ans.T[0]
the2_arr = ans.T[2]

# 4. Correct Coordinate Conversion for Animation
# Define a numerical function using lambdify on the symbolic expressions
# Crucially, we use the symbolic variables L1, L2, the1, the2 as inputs to lambdify,
# and specify 'numpy' as the backend.
get_coords = smp.lambdify((L1, L2, the1, the2),
                          (x1_smp, y1_smp, x2_smp, y2_smp),
                          'numpy')

# Calculate the numerical coordinates using the numerical constants and results
x1, y1, x2, y2 = get_coords(L1_num, L2_num, the1_arr, the2_arr)

# 5. Animate
def animate(i):
    # Set the data for the double pendulum
    # The line segments are [anchor to m1], [m1 to m2]
    ln1.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])

# Setup plot
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.set_facecolor('#0d1117') # Dark background for better visualization
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
ax.set_title("Double Pendulum Chaos Simulation", color='white')

# Plot the line (ro--: red color, circle markers, dashed line)
# lw=3 (line width), markersize=8
ln1, = plt.plot([], [], 'o-', color='#00aaff', lw=3, markersize=8, markeredgecolor='white', markerfacecolor='#ff4757')

# Set limits
ax.set_ylim(-(L1_num + L2_num + 0.5), (L1_num + L2_num + 0.5))
ax.set_xlim(-(L1_num + L2_num + 0.5), (L1_num + L2_num + 0.5))
ax.set_aspect('equal') # Important to make the plot look correct

# Create the animation
# frames=1001 to use all data points in t_num
# interval=40 (25 fps since 1000/40 = 25)
ani = animation.FuncAnimation(fig, animate, frames=len(t_num), interval=40)

# Save the animation as a GIF
# fps=25 matches the interval speed
print("Saving animation to pen.gif...")
ani.save('pen.gif', writer='pillow', fps=25)
print("Animation saved successfully.")







