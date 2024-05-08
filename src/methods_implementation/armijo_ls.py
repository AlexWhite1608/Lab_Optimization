from . import gradient
import numpy as np

class ArmijoLS(gradient.GradientDescentMethod):

    def __init__(self,  delta_k=0.5, delta=0.5, gamma=0.5, max_iter=1000):
        super().__init__(max_iter=max_iter)
        self._delta_k = delta_k
        self._delta = delta
        self._gamma = gamma
        self._name = 'Armijo Line Search'

    def optimize(self):
        x = self._problem.x0
        for i in range(self._max_iter):
            gradient = self._problem.grad(x)
            if np.linalg.norm(gradient) < self._tol:
                break
            step_size = self.__armijo_ls(x, gradient, self._delta_k, self._delta, self._gamma)
            x = x - step_size * gradient
        
            print(f"{self._name}: {i+1}; (x,y): {x}")

        print("----------------------------------------------\n")
        return self._problem.obj(x), i + 1

    def __armijo_ls(self, x, gradient, delta_k, delta, gamma):
        alpha = delta_k
        while self._problem.obj(x - alpha * gradient) > self._problem.obj(x) - gamma * alpha * np.linalg.norm(gradient)**2:
            alpha *= delta

        return alpha