import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# Symbolic variables
x1, x2 = sp.symbols('x1 x2')

# Define the symbolic function f(x)
f_sym = 10 * (x2 - x1**2)**2 + (1 - x1)**2

# Define the regular function f(x)
def f(x):
    x1, x2 = x
    return 10 * (x2 - x1**2)**2 + (1 - x1)**2

x = [x1, x2]
grad_sym = sp.Matrix([sp.diff(f_sym, var) for var in x])
hess_sym = sp.hessian(f_sym, x)
grad_func = sp.lambdify(x, grad_sym, 'numpy')
hess_func = sp.lambdify(x, hess_sym, 'numpy')

# Compute Cauchy point
def cauchy_point(gk, Bk, Delta):
    norm_g = np.linalg.norm(gk)
    g_B_g = gk.T @ Bk @ gk

    if g_B_g <= 0:
        tau_k = 1
    else:
        tau_k = min(1, (norm_g**3) / (Delta * g_B_g))
    
    pk_C = -tau_k * (Delta * gk / norm_g)
    return pk_C

# Initial values
# xk = np.array([0, -1])
xk = np.array([0, 0.5])
gk = np.array(grad_func(*xk), dtype=float).flatten()
Bk = np.array(hess_func(*xk), dtype=float)

# Compute p and ||p||
Deltas = np.linspace(0.01, 2, 10)
p_values = np.array([cauchy_point(gk,Bk,Delta) for Delta in Deltas])
p_norms = np.array([np.linalg.norm(cauchy_point(gk,Bk,Delta)) for Delta in Deltas])

# Plot ||p|| vs Delta
plt.figure(figsize=(6, 5))
plt.plot(Deltas, p_norms, 'o-', label=r"$||p_k^C||$ (Cauchy Point)")
plt.xlabel(r"$\Delta$")
plt.ylabel(r"$||p_k^C||$")
plt.title(r"$||P_k^C||$ vs. Trust Region Radius at x = (0, 0.5)")
plt.legend()
plt.grid()
plt.show()

# Plot p vs. trust region radii
plt.scatter(xk[0], xk[1], color='blue', label='Point (0, 0.5)', s=4)
for r, p in zip(Deltas, p_values):
    circle = plt.Circle(xk, r, color='blue', fill=False, linestyle='-',
                        label='trust region radius' if r == Deltas[0] else "")
    plt.gca().add_artist(circle)
    plt.quiver(xk[0], xk[1], p[0], p[1], angles='xy', scale_units='xy', scale=1, color='red', width=0.004)
plt.xlim(-1., 2.)
plt.ylim(-2, 1)
# plt.ylim(-1.6,1.6)
plt.title('Trust Region vs. Cauchy Optimal Steps at x = (0, 0.5)')
plt.xlabel('x1')
plt.ylabel('x2')
plt.axhline(0, color='black', linewidth=0.5, ls='--')
plt.axvline(0, color='black', linewidth=0.5, ls='--')
plt.grid()
plt.legend()
plt.show()