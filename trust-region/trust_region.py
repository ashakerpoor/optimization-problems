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
