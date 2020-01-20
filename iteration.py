import numpy as np
from numpy import linalg as LA
from scipy import optimize

# def powerIteration(transitionWeights, rsp=0.15, epsilon=0.00001, maxIterations=1000):
#     # Power iteration:
#     for iteration in range(maxIterations):
#         oldState = state.copy()
#         state = state.dot(transitionProbabilities)
#         delta = state - oldState
#         if __euclideanNorm(delta) < epsilon:
#             break
#     return state


if __name__ == '__main__':
    def func(x, c1, c2):
        return np.sqrt(c1 / (x + c2))

    c1 = np.array([10, 12.])
    c2 = np.array([3, 5.])
    optimize.fixed_point(func, [1.2, 1.3], args=(c1, c2))
