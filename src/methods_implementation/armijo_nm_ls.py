from . import gradient
import numpy as np

class ArmijoNMLS(gradient.GradientDescentMethod):

    def __init__(self,  delta_k=0.5, delta=0.5, gamma=0.5, max_iter=1000):
        super().__init__(max_iter=max_iter)
        self._delta_k = delta_k
        self._delta = delta
        self._gamma = gamma
        self._name = 'Armijo Non Monotone Line Search'

    def optimize(self):
        x = self._problem.x0
        x_seq = [] 

        for i in range(self._max_iter):
            gradient = self._problem.grad(x)
            if np.linalg.norm(gradient) < self._tol:
                break
            W = self.__get_W(x, i, x_seq)
            step_size = self.__armijo_nm(x, gradient, W)

            x = x - step_size * gradient

            print(f"{self._name}: {i+1}; (x,y): {x}")

        print("----------------------------------------------\n")
        return x, i + 1
    
    def __armijo_nm(self, x, gradient, W):
        alpha = self._delta_k
        j=0

        while self._problem.obj(x - alpha * gradient) > W - self._gamma * alpha * np.dot(gradient, gradient):
            alpha *= self._delta
            j += 1

        return alpha
    
    def __get_W(self, x, i, x_seq, M=3):
        x_seq.append(x)
        W = 0

        # Calcolo delle sequenze degli f(x)
        if i == 0:
            values = [self._problem.obj(x_seq[0])]
        else:
            values = [self._problem.obj(x_seq[i - j]) for j in range(1, min(i, M) + 1)]

        W = np.max(values)

        return W