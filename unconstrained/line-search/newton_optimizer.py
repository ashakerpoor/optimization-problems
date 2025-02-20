import sympy as sp
import numpy as np


def rosenbrock(x, y):   # Rosenbrock function
    return 100 * (y - x**2)**2 + (1 - x)**2


def gradient(f, variable_list):
    """
    Calculate the gradient of a multivariate function.

    Parameters:
    f (sympy expression): A multivariate function.
    variable_list (list of sympy symbols): A list of variables as sympy symbols.

    Returns:
    numpy.ndarray: A numpy array of partial derivatives of the function.
    """
    derivatives_array = []
    
    for variable in variable_list:
        derivative = sp.diff(f, variable)
        derivatives_array.append(derivative)
    
    return np.array(derivatives_array)


def hessian(f, variable_list):
    """
    Calculate the Hessian matrix of a multivariate function.

    Returns:
    numpy.ndarray: The Hessian matrix in the form of a numpy array.
    """
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

    f = rosenbrock(x1, x2)
    grad = gradient(f, variables)
    hess = hessian(f, variables)

    print(f"Gradient: {grad}\nHessian: {hess}")