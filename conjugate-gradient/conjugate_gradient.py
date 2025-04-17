import numpy as np


def linear_cg(A, b, x0, iter_bound, tol=1e-6):
    """
    Implements the linear version of the conjugate gradient method.

    A: coefficient matrix, a symmetric positive-definite matrix
    b: constant (right-hand side) vector
    x0: initial iterate
    """

    k = 0
    n = b.shape[0]
    x = np.full((n,1), float(x0)) if np.isscalar(x0) else np.array(x0, dtype=float).reshape(n,1)  # Ensure x is (n,1)
    r_k = A @ x - b
    p = -r_k.copy().reshape(n,1)

    while (k < iter_bound):
        if (np.linalg.norm(r_k) < tol):
            print(f'Computation converged at iteration = {k}.')
            break

        Ap = A @ p
        alpha = ((r_k.T @ r_k) / (p.T @ Ap))[0,0]
        x += alpha * p
        r_k1 = r_k + alpha * Ap

        if np.linalg.norm(r_k1) < tol:
            print(f'Converged at iteration {k+1}.')
            break

        beta = ((r_k1.T @ r_k1) / (r_k.T @ r_k))[0,0]
        p = -r_k1 + beta * p

        r_k = r_k1
        k += 1

    return x    # Approximate solution


if __name__ == "__main__":
    n = 20   # This is the dimension of the linear system
    x0 = 0
    b = np.ones(n).reshape(n, 1)
    A = np.array([[1 / (i + j - 1) for j in range(1, n+1)] for i in range(1, n+1)])
    # print("Coefficient matrix:\n", A)

    x = linear_cg(A, b, x0, iter_bound=1000, tol=1e-6)
    print("Solution x:\n", x)
