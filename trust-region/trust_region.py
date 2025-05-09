import sympy as sp
import numpy as np



# Trust region method 
def tr(f, variables, radius_bound, iterations, init_point=[]):
    iterations = int(iterations)
    xT = np.array(init_point, dtype=float)
    grad = gradient(f, variables)
    hess = hessian(f, variables)
    f_numeric = sp.lambdify(variables, f, 'numpy')

    radius = 0.5 * radius_bound
    eta = 0.125  # eta \in [0, 0.25)

    in_file = open("TR_output.dat", "w")

    for k in range(iterations):
        # evaluated_grad = np.array([g.subs(dict(zip(variables, xT))) for g in grad], dtype=float)
        evaluated_grad = np.array([g(*xT) for g in grad], dtype=float)
        # evaluated_hess = np.array([[h.subs(dict(zip(variables, xT))) for h in row] for row in hess], dtype=float)
        evaluated_hess = np.array([[(h(*xT)) for h in row] for row in hess])

        # Obtain pk (step direction) using the dogleg method
        p_k = step_direction(evaluated_grad, evaluated_hess, radius)
        pk_norm = np.linalg.norm(p_k)

        rho = reduction_ratio(f_numeric, xT, evaluated_grad, evaluated_hess, p_k)

        if rho < 0.25:
            radius *= 0.25
        elif rho > 0.75 and pk_norm == radius:
            radius = min(2 * radius, radius_bound)

        f_evaluated = f_numeric(*xT)
        grad_norm = np.linalg.norm(evaluated_grad)

        in_file.write("%10d %10.5f %10.5f\n" % (k, float(f_evaluated), float(grad_norm)))

        if rho > eta:
            xT += p_k

        # Convergence check
        if grad_norm < 1e-6:
            print(f'Computation converged.\nx^* = {xT}, f(x^*) = {f_numeric(*xT)}, ||gradient|| = {grad_norm}')
            break

    in_file.close()

    return xT


def cost_function(f_k, g_k, B_k, p):
    """
    Quadratic approximation of the function at x_k.

    Parameters:
    f_k: function value at x_k
    g: evaluated gradient at x_k
    B: evaluated Hessian at x_k
    p: step direction
    """
    return f_k + np.dot(g_k, p) + 0.5 * np.dot(p, np.dot(B_k, p))


def reduction_ratio(f, x_k, g_k, B_k, p):
    """
    Parameters:
    variables: array of the function's f variables
    x_k: current point
    g_k: evaluated gradient at xk
    B_k: evaluated hessian at xk
    p: step direction
    """

    x_k1 = x_k + p
    f_k = f(*x_k) # Evaluate f at x_k
    f_k1 = f(*x_k1) # Evaluate f at x_k + p

    n = len(p)
    m_k0 = cost_function(0, g_k, B_k, np.zeros(n))  # Model at p=0
    m_kp = cost_function(0, g_k, B_k, p)    # Model at p=p_k

    return (f_k - f_k1) / (m_k0 - m_kp) if (m_k0 - m_kp) != 0 else 1.0

    
def step_direction(g_k, B_k, radius):
    """Compute trust region step using the dogleg method."""
    
    # Cauchy step
    pU = - (np.dot(g_k, g_k) / np.dot(g_k, np.dot(B_k, g_k))) * g_k   
    # full Newton step
    pB = -np.linalg.solve(B_k, g_k)

    if np.linalg.norm(pB) <= radius and is_pd(B_k):
        return pB
    elif np.linalg.norm(pU) >= radius:
        # return (radius / np.linalg.norm(pU)) * pU
        return -radius * (g_k / np.linalg.norm(g_k))
    else:
        dp = pB - pU
        a = np.dot(dp, dp)
        b = 2 * np.dot(pU, dp)
        c = np.dot(pU, pU) - radius**2
        
        # Solution to ||pU + (tau-1)(pB-pU)||^2 = Delta^2
        # This is tau - 1
        tau = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
        return pU + tau * dp


def rosenbrock(x, y):   # Rosenbrock function
    return 100 * (y - x**2)**2 + (1 - x)**2


def gradient(f, variable_list):
    """
    Parameters:
    f (sympy expression): A multivariate function.
    variable_list (list of sympy symbols): A list of variables as sympy symbols.

    Returns:
    A list of Numpy expressions
    """
    
    return [sp.lambdify(variable_list,sp.diff(f, variable), 'numpy') for variable in variable_list]


def hessian(f, variable_list):
    """
    Returns:
    numpy.ndarray: The Hessian matrix in the form of a numpy expressions.
    """
    n = len(variable_list)
    derivatives_matrix = np.zeros((n, n), dtype=object)

    for i, var1 in enumerate(variable_list):
        first_deriv = sp.diff(f, var1)
        for j, var2 in enumerate(variable_list):
            second_deriv = sp.diff(first_deriv, var2)
            derivatives_matrix[i, j] = sp.lambdify(variable_list, second_deriv, 'numpy')

    return derivatives_matrix
    
    
# Check for positive definiteness
def is_pd(B_k):
    try:
        np.linalg.cholesky(B_k)
        return True
    except np.linalg.LinAlgError:
        return False
        
        
if __name__ == "__main__":
    x1,x2 = sp.symbols("x1 x2")
    variables = [x1, x2]
    # init_point=[1.2, 1.2]
    init_point=[-1.2,1.0]
    # iterations = 1000000
    iterations = 100
    radius_bound = 2.

    f_rosenbrock = rosenbrock(x1, x2)
    minimizer = tr(f_rosenbrock, variables, radius_bound, iterations, init_point)
    print("Minimizer point:", minimizer)
