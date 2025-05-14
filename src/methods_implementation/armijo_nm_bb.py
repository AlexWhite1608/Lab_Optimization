from . import gradient
import numpy as np

class ArmijoNMBB(gradient.GradientDescentMethod):

    def __init__(self, technique, armijo_nm_params={}, sigma_a=1e-4, sigma_b=1e-4, M=3):
        super().__init__()
        self._technique = technique
        self._armijo_nm_params = armijo_nm_params
        self._sigma_a = sigma_a
        self._sigma_b = sigma_b
        self._M = M
        self._name = 'Barzilai-Borwein Line Search'

    def optimize(self):
        k = 0
        x = self._problem.x0
        self._x_history = [x]
        self._obj_history.append(self._problem.obj(x))

        
        x_k_minus_1 = x
        grad_k_minus_1 = self.evaluate_gradient(x)

        print(f"{self._name}: {self._problem.name}, starting point: {x}")

        while k < self._max_iter:
            gradient = self.evaluate_gradient(x)
            norm_grad = np.linalg.norm(gradient)

            if norm_grad < self._tol:
                break

            s = x - x_k_minus_1
            y = gradient - grad_k_minus_1

            mu = self.__get_mu(s, y, k)
            d_k = -gradient / mu

            alpha = self.__armijo_nm(x, d_k)

            x_k_minus_1 = x
            grad_k_minus_1 = gradient

            x = x + alpha * d_k

            self._x_history.append(x)
            self._obj_history.append(self._problem.obj(x))

            k += 1

            print(f"{self._name}: Iteration {k}; x: {x}; Objective: {self._problem.obj(x)}")

        return self._problem.obj(x), k

    def __get_mu(self, s, y, k):
        epsilon = 1e-8
        s_dot_s = np.dot(s, s) + epsilon
        s_dot_y = np.dot(s, y) + epsilon
        y_dot_y = np.dot(y, y) + epsilon

        if self._technique == 'A':
            return s_dot_y / s_dot_s
        elif self._technique == 'B':
            return y_dot_y / s_dot_y
        elif self._technique == 'AB':
            if k % 2 == 0:
                return s_dot_y / s_dot_s
            else:
                return y_dot_y / s_dot_y
        else:
            raise ValueError("Invalid mu technique")

    def __armijo_nm(self, x, d_k):
        alpha = self._armijo_nm_params.get('delta_k', 0.3)
        delta = self._armijo_nm_params.get('delta', 0.4)
        gamma = self._armijo_nm_params.get('gamma', 0.6)
        W = self.__get_W()

        while self.evaluate_objective(x + alpha * d_k) > W - gamma * alpha * np.dot(d_k, d_k):
            alpha *= delta

        return alpha

    def __get_W(self):
        values = [self._problem.obj(x) for x in self._x_history[-self._M:]]
        return max(values) if values else self.evaluate_objective(self._x_history[-1])
