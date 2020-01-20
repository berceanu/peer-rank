import numpy as np
from numpy import linalg as LA
from scipy import optimize

if __name__ == '__main__':

    def rhs(X, A, alpha):
        return (1 - alpha) * X + alpha / np.sum(X) * np.dot(A, X)

    m = 10  # number of agents
    # A = np.identity(m)  # grade matrix, shape m x m
    A = np.full((m, m), 0.8)
    alpha = 0.1  # convergence parameter

    X0 = np.dot(A, np.full(m, 1/m))
    X = optimize.fixed_point(rhs, X0, args=(A, alpha))
    print(X)

    w, v = LA.eig(A)
    # A.X = np.sum(X)*X


