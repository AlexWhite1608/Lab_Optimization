from . import gradient
import numpy as np

class ConstantStepSize(gradient.GradientDescentMethod):
    
    def __init__(self, parameters):
        super().__init__()
        self._learning_rate = parameters.get('learning_rate')
        self._name = 'Constant Step Size'

    def optimize(self):
        x = self._problem.x0

        print(f"{self._name}: {self._problem.name}")

        for i in range(self._max_iter):
            gradient = self._problem.grad(x)
            if np.linalg.norm(gradient) < self._tol:
                break
            x = x - self._learning_rate * gradient

            self._x_history.append(x)
            self._obj_history.append(self._problem.obj(x))

            print(f"{self._name}: {i+1}; (x,y): {x}")

        print("----------------------------------------------\n")

        super().plot_results()

        self._x_history = []  
        self._obj_history = [] 

        return self._problem.obj(x), i + 1