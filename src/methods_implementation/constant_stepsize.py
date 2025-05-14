from . import gradient
import numpy as np

class ConstantStepSize(gradient.GradientDescentMethod):
    
    def __init__(self, parameters):
        super().__init__()
        self._learning_rate = parameters.get('learning_rate')
        self._name = 'Constant Step Size'

    def optimize(self):
        x = self._problem.x0
        self._obj_history.append(self._problem.obj(x))


        print(f"{self._name}: {self._problem.name}, starting point: {x}")

        for i in range(self._max_iter):
            gradient = self.evaluate_gradient(x)
            if np.linalg.norm(gradient) < self._tol:
                break
            x = x - self._learning_rate * gradient

            self._x_history.append(x)
            self._obj_history.append(self._problem.obj(x))
            self._objective_evaluations += 1

            print(f"{self._name}: Iteration {i}; x: {x}; Objective: {self._problem.obj(x)}")

        print("----------------------------------------------\n")

        #self._x_history = []  
        #self._obj_history = [] 

        return self._problem.obj(x), i + 1