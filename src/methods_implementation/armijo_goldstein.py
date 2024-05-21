from . import gradient
import numpy as np

class ArmijoGoldstein(gradient.GradientDescentMethod):

    def __init__(self, parameters):
        super().__init__()
        self._delta_k = parameters.get('delta_k')
        self._delta = parameters.get('delta')
        self._gamma1 = parameters.get('gamma1')
        self._gamma2 = parameters.get('gamma2')
        self._name = 'Armijo-Goldstein Line Search'

    def optimize(self):
        x = self._problem.x0

        print(f"{self._name}: {self._problem.name}")

        for i in range(self._max_iter):
            gradient = self._problem.grad(x)
            if np.linalg.norm(gradient) < self._tol:
                break
            step_size = self.__armijo_goldstein_ls(x, gradient, self._delta_k, self._delta, self._gamma1, self._gamma2)

            x = x - step_size * gradient

            self._x_history.append(x)
            self._obj_history.append(self._problem.obj(x))

            print(f"{self._name}: {i+1}; (x,y): {x}")

        print("----------------------------------------------\n")

        super().plot_results()

        self._x_history = []  
        self._obj_history = [] 

        return self._problem.obj(x), i + 1

    def __armijo_goldstein_ls(self, x, gradient, delta_k, delta, gamma1, gamma2):
        alpha = delta_k
        while self._problem.obj(x - alpha * gradient) > self._problem.obj(x) - gamma1 * alpha * np.linalg.norm(gradient)**2:
            alpha *= delta

        if alpha < delta_k:
            return alpha
        
        # Condizioni di Goldstein
        goldstein_condition_1 = self._problem.obj(x - alpha * gradient) < self._problem.obj(x) - gamma2 * alpha * np.linalg.norm(gradient)**2 
        goldstein_condition_2 = self._problem.obj(x - (alpha/delta) * gradient) < np.min([self._problem.obj(x - alpha * gradient), self._problem.obj(x) + gamma1 * (alpha/delta)*np.linalg.norm(gradient)**2])

        while goldstein_condition_1 and goldstein_condition_2:
            alpha /= delta
        
        return alpha