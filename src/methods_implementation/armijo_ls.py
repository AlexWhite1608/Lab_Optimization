from . import gradient
import numpy as np

class ArmijoLS(gradient.GradientDescentMethod):

    def __init__(self, parameters):
        super().__init__()
        self._delta_k = parameters.get('delta_k')
        self._delta = parameters.get('delta')
        self._gamma = parameters.get('gamma')
        self._name = 'Armijo Line Search'

    def optimize(self):
        x = self._problem.x0

        print(f"{self._name}: {self._problem.name}")

        for i in range(self._max_iter):
            gradient = self._problem.grad(x)
            if np.linalg.norm(gradient) < self._tol:
                break
            step_size = self.__armijo_ls(x, gradient, self._delta_k, self._delta, self._gamma)
            
            x = x - step_size * gradient
        
            self._x_history.append(x)
            self._obj_history.append(self._problem.obj(x))

            print(f"{self._name}: {i+1}; (x,y): {x}")

        print("----------------------------------------------\n")

        super().plot_results()

        self._x_history = []  
        self._obj_history = [] 

        return self._problem.obj(x), i + 1

    def __armijo_ls(self, x, gradient, delta_k, delta, gamma):
        alpha = delta_k
        while self._problem.obj(x - alpha * gradient) > self._problem.obj(x) - gamma * alpha * np.linalg.norm(gradient)**2:
            alpha *= delta

        return alpha