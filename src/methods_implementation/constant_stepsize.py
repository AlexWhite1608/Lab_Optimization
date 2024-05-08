from . import gradient
import numpy as np

class ConstantStepSize(gradient.GradientDescentMethod):
    
    def __init__(self, learning_rate=0.01, max_iter=1000):
        super().__init__(max_iter=max_iter)
        self._learning_rate = learning_rate
        self._name = 'Constant Step Size'

    def optimize(self):
        x = self._problem.x0
        for i in range(self._max_iter):
            gradient = self._problem.grad(x)
            if np.linalg.norm(gradient) < self._tol:
                break
            x = x - self._learning_rate * gradient

            print(f"{self._name}: {i+1}; (x,y): {x}")

        print("----------------------------------------------\n")
        return x, i + 1