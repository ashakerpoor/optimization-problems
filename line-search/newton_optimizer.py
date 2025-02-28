import sympy as sp
import numpy as np


def newton(f, variable_list, iterations, init_point=[]):
    iterations = int(iterations)
    xT = np.array(init_point, dtype=float)
    grad = gradient(f, variable_list)
    hess = hessian(f, variable_list)
    
    f_numeric = sp.lambdify(variable_list, f, 'numpy')

    in_file = open('output.dat', 'w')
    
    for i in range(iterations):
        evaluated_grad = np.array([derivative.subs(dict(zip(variable_list, xT))) 
                                   for derivative in grad], dtype=float)
        evaluated_hess = np.array([[entry.subs(dict(zip(variable_list, xT))) for entry in row] 
                                   for row in hess], dtype=float)

        grad_norm = np.linalg.norm(evaluated_grad)

        step_direction = -np.linalg.solve(evaluated_hess, evaluated_grad)
        alpha = line_search(f_numeric, 0.5, 1e-4, evaluated_grad, xT, step_direction, alpha=1.0)

        if i%1 == 0:
            print(f'{i} step length: {alpha}')
            f_evaluated = f_numeric(*xT)
            in_file.write("%10d %10.5f %10.5f\n" % (i, float(f_evaluated), float(grad_norm)))

        # Newton update step
        xT += alpha * step_direction

        if grad_norm < 1e-6:
            print(f'Computation converged.\nx^* = {xT}, f(x^*) = {f_numeric(*xT)}, ||gradient|| = {grad_norm}')
            break
    
    in_file.close()

    return xT


def line_search(f_numeric, rho, c, grad, x_k, step_direction, alpha=1.0):
    f_k = f_numeric(*x_k)
    
    while True:
        x_k1 = x_k + alpha * step_direction
        f_k1 = f_numeric(*x_k1)
        
        # Armijo condition
        if f_k1 <= f_k + c * alpha * np.dot(grad, step_direction):
            break
        
        # Reduce step size
        alpha *= rho

    return alpha


def rosenbrock(x, y):   # Rosenbrock function
    return 100 * (y - x**2)**2 + (1 - x)**2


def gradient(f, variable_list):
    """
    Parameters:
    f (sympy expression): A multivariate function.
    variable_list (list of sympy symbols): A list of variables as sympy symbols.
    """
    derivatives_array = []
    
    for variable in variable_list:
        derivative = sp.diff(f, variable)
        derivatives_array.append(derivative)
    
    return np.array(derivatives_array)


def hessian(f, variable_list):
    n = len(variable_list)
    derivatives_matrix = np.zeros((n, n), dtype=object)

    for i, var1 in enumerate(variable_list):
        first_deriv = sp.diff(f, var1)
        for j, var2 in enumerate(variable_list):
            second_deriv = sp.diff(first_deriv, var2)
            derivatives_matrix[i, j] = second_deriv

    return derivatives_matrix



if __name__ == "__main__":
    x1,x2 = sp.symbols("x1 x2")
    variables = [x1, x2]
    init_point=[1.2, 1.2]
    # init_point=[-1.2,1.0]
    iterations = 1000000

    f_rosenbrock = rosenbrock(x1, x2)
    optimal_point = newton(f_rosenbrock, variables, iterations, init_point)
    print("Optimal point:", optimal_point)