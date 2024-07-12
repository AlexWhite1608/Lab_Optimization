from . import gradient
import numpy as np

class ArmijoNMBB(gradient.GradientDescentMethod):

    def __init__(self, technique, armijo_goldstein_params={}, armijo_nm_params={}, sigma_a=1e-4, sigma_b=1e-4, M=15, N=10):
        super().__init__()
        self._technique = technique
        self._armijo_goldstein_params = armijo_goldstein_params
        self._armijo_nm_params = armijo_nm_params
        self._sigma_a = sigma_a
        self._sigma_b = sigma_b
        self._M = M
        self._N = N
        self._name = 'Barzilai-Borwein Line Search'

    def optimize(self):
        k = 0
        x = self._problem.x0
        self._x_history = [x]
        self._obj_history = [self._problem.obj(x)]
        
        x_k_minus_1 = x
        grad_k_minus_1 = self.evaluate_gradient(x)

        while k < self._max_iter:
            gradient = self.evaluate_gradient(x)
            norm_grad = np.linalg.norm(gradient)

            if norm_grad < self._tol:
                break

            s = x - x_k_minus_1
            y = gradient - grad_k_minus_1

            # Per evitare divisione per zero si aggiunge un epsilon
            epsilon = 1e-8
            mu = self.__get_mu(s, y, k, epsilon)
            if mu == 0 or np.isnan(mu):
                mu = self.__armijo_goldstein_ls(x, gradient)

            d_k = -gradient / (mu + epsilon)  

            alpha = self.__armijo_nm(x, gradient, d_k)

            x_k_minus_1 = x
            grad_k_minus_1 = gradient

            x = x + alpha * d_k

            self._x_history.append(x)
            self._obj_history.append(self._problem.obj(x))

            k += 1

            print(f"{self._name}: Iteration {k}; x: {x}; Objective: {self._problem.obj(x)}")

        return self._problem.obj(x), k

    def __get_mu(self, s, y, k, epsilon):
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

    def __armijo_nm(self, x, gradient, d_k):
        alpha = self._armijo_nm_params.get('delta_k', 0.1)
        delta = self._armijo_nm_params.get('delta', 0.5)
        gamma = self._armijo_nm_params.get('gamma', 0.9)
        W = self.__get_W(x)

        while self._problem.obj(x + alpha * d_k) > W - gamma * alpha * np.dot(gradient, d_k):
            alpha *= delta

        return alpha

    def __get_W(self, x):
        values = [self._problem.obj(x) for x in self._x_history[-self._M:]]
        return max(values) if values else self._problem.obj(x)

    def __armijo_goldstein_ls(self, x, gradient):
        delta_k = self._armijo_goldstein_params.get('delta_k', 0.1)
        delta = self._armijo_goldstein_params.get('delta', 0.5)
        gamma1 = self._armijo_goldstein_params.get('gamma1', 0.1)
        gamma2 = self._armijo_goldstein_params.get('gamma2', 0.9)
        alpha = delta_k

        while self._problem.obj(x - alpha * gradient) > self._problem.obj(x) - gamma1 * alpha * np.linalg.norm(gradient)**2:
            alpha *= delta

        while self._problem.obj(x - alpha * gradient) < self._problem.obj(x) - gamma2 * alpha * np.linalg.norm(gradient)**2:
            alpha /= delta

        return alpha
