from . import gradient
import numpy as np

class WolfeLS(gradient.GradientDescentMethod):

    def __init__(self,  gamma, sigma, alpha_l=0, alpha_u=100, max_iter=1000):
        super().__init__(max_iter=max_iter)
        self._gamma = gamma
        self._sigma = sigma
        self._alpha_l = alpha_l
        self._alpha_u = alpha_u
        self._name = 'Wolfe Line Search'

    def optimize(self):

        x = self._problem.x0
        for i in range(self._max_iter):
            gradient = self._problem.grad(x)
            if np.linalg.norm(gradient) < self._tol:
                break
            step_size = self.__wolfe_ls(x, gradient)

            x = x - step_size * gradient

            print(f"{self._name}: {i+1}; (x,y): {x}")

        print("----------------------------------------------\n")
        return x, i + 1
    
    def __wolfe_ls(self, x, gradient):
        while True:
            alpha = np.random.uniform(self._alpha_l, self._alpha_u)
            
            obj_alpha = self._problem.obj(x - alpha * gradient)
            grad_alpha = np.linalg.norm(self._problem.grad(x - alpha * gradient)) ** 2
            
            # Condizioni di Wolfe forti
            wolfe_1 = obj_alpha <= self._problem.obj(x) - self._gamma * alpha * np.dot(gradient, gradient)
            wolfe_2 = grad_alpha <= self._sigma * np.dot(gradient, gradient)
            
            if wolfe_1 or wolfe_2:
                return alpha
            
            if obj_alpha > self._problem.obj(x) - self._gamma * alpha * np.dot(gradient, gradient):
                self._alpha_u = alpha
            elif obj_alpha <= self._problem.obj(x) - self._gamma * alpha * np.dot(gradient, gradient) and grad_alpha < self._sigma * np.dot(gradient, gradient):
                self._alpha_l = alpha
            elif obj_alpha <= self._problem.obj(x) - self._gamma * alpha * np.dot(gradient, gradient) and grad_alpha > self._sigma * np.dot(gradient, gradient):
                self._alpha_u = alpha

    