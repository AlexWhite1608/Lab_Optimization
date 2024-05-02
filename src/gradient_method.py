import numpy as np
import matplotlib.pyplot as plt

class GradientType:

    """
    Class that contains the gradient descent variants.

    Attributes
    ----------
    CONSTANT_STEPSIZE : int
        gradient descent with constant stepsize

    ARMIJO_LS : int
        gradient descent with Armijo line-search;

    ARMIJO_GOLDSTEIN : int
        gradient descent with Armijo line-search with expansion (Armijo-Goldstein)

    WOLFE_LS : int
        gradient descent with Wolfe line-search

    ARMIJO_NON_MONOTONE : int
        gradient descent with nonmonotone Armijo line-search

    BARZILAI_BORWEIN : int
        gradient descent with nonmonotone Armijo line-search, with initial stepsize
        set by the Barzilai-Borwein rule

    """

    CONSTANT_STEPSIZE = 1
    ARMIJO_LS = 2
    ARMIJO_GOLDSTEIN = 3
    WOLFE_LS = 4
    ARMIJO_NON_MONOTONE = 5
    BARZILAI_BORWEIN = 6

class GradientDescentMethod:

    """ def __init__(self, method_type):
        self.type = method_type """

    """ def gradient_descent(self, problem, type, tol=1e-6, max_iter=1000):

        # problem parameters
        x_0 = problem.x0
        f = problem.obj(x)

        x = x_0
        for i in range(max_iter):
            gradient = problem.grad(x)
            if np.linalg.norm(gradient) < tol:
                break
            x = self.__method(x, gradient, type)

        return x, i + 1
    
    def __method(self, x, gradient, type: GradientType):
        if(type == GradientType.CONSTANT_STEPSIZE):
            pass
        elif(type == GradientType.ARMIJO_LS):
            pass
        elif(type == GradientType.ARMIJO_GOLDSTEIN):
            pass
        elif(type == GradientType.WOLFE_LS):
            pass
        elif(type == GradientType.ARMIJO_NON_MONOTONE):
            pass
        elif(type == GradientType.BARZILAI_BORWEIN):
            pass """

    def __init__(self, problem, tol=1e-6, max_iter=1000):
        
        # problem parameters
        self._problem = problem
        self._x0 = problem.x0
        self._f = problem.obj(self._x0)

        # convergence settings
        self._tol = tol
        self._max_iter = max_iter
        pass

    # ALGORITMO GRADIENTE COSTANTE
    def gd_const(self, learning_rate=0.01):
        x = self._x0
        for i in range(self._max_iter):
            gradient = self._problem.grad(x)
            if np.linalg.norm(gradient) < self._tol:
                break
            x = x - learning_rate * gradient

            print(f"Iteration {i+1}, x: {x}")

        return x, i + 1
    
    # ALGORITMO LS ARMIJO
    def gd_armijo(self, delta_k=0.5, delta=0.5, gamma=0.5):
        x = self._x0
        for i in range(self._max_iter):
            gradient = self._problem.grad(x)
            if np.linalg.norm(gradient) < self._tol:
                break
            step_size = self.__armijo_ls(x, gradient, delta_k, delta, gamma)

            x = x - step_size * gradient

            print(f"Iteration {i+1}, (x,y): {x}")

        return x, i + 1
    
    # ALGORITMO LS ARMIJO-GOLDSTEIN
    def gd_goldstein(self, delta_k=0.5, delta=0.5, gamma1=0.5, gamma2=0.5):
        x = self._x0
        for i in range(self._max_iter):
            gradient = self._problem.grad(x)
            if np.linalg.norm(gradient) < self._tol:
                break
            step_size = self.__armijo_goldstein_ls(x, gradient, delta_k, delta, gamma1, gamma2)

            x = x - step_size * gradient

            print(f"Iteration {i+1}, (x,y): {x}")

        return x, i + 1
    
    # ALGORITMO LS WOLFE (ALGW2)
    def gd_wolfe(self, gamma, sigma):
        x = self._x0
        for i in range(self._max_iter):
            gradient = self._problem.grad(x)
            if np.linalg.norm(gradient) < self._tol:
                break
            step_size = self.__wolfe_ls(x, gradient, gamma, sigma)

            x = x - step_size * gradient

            print(f"Iteration {i+1}; (x,y): {x}")

        return x, i + 1
    
    def gd_armijo_non_monotone(self, delta_k=0.5, delta=0.5, gamma=0.5):
        pass
    
    # Wolfe LineSearch
    def __wolfe_ls(self, x, gradient, gamma, sigma, alpha_l=0, alpha_u=100):
        while True:
            alpha = np.random.uniform(alpha_l, alpha_u)
            
            obj_alpha = self._problem.obj(x - alpha * gradient)
            grad_alpha = np.linalg.norm(self._problem.grad(x - alpha * gradient)) ** 2
            
            # Condizioni di Wolfe forti
            wolfe_1 = obj_alpha <= self._problem.obj(x) - gamma * alpha * np.dot(gradient, gradient)
            wolfe_2 = grad_alpha <= sigma * np.dot(gradient, gradient)
            
            if wolfe_1 or wolfe_2:
                return alpha
            
            if obj_alpha > self._problem.obj(x) - gamma * alpha * np.dot(gradient, gradient):
                alpha_u = alpha
            elif obj_alpha <= self._problem.obj(x) - gamma * alpha * np.dot(gradient, gradient) and grad_alpha < sigma * np.dot(gradient, gradient):
                alpha_l = alpha
            elif obj_alpha <= self._problem.obj(x) - gamma * alpha * np.dot(gradient, gradient) and grad_alpha > sigma * np.dot(gradient, gradient):
                alpha_u = alpha

    # Armijo LineSearch 
    def __armijo_ls(self, x, gradient, delta_k, delta, gamma):
        alpha = delta_k
        while self._problem.obj(x - alpha * gradient) > self._problem.obj(x) - gamma * alpha * np.linalg.norm(gradient)**2:
            alpha *= delta

        return alpha

    # Armijo-Goldstein LineSearch 
    def __armijo_goldstein_ls(self, x, gradient, delta_k, delta, gamma1, gamma2):
        alpha = delta_k
        while self._problem.obj(x - alpha * gradient) > self._problem.obj(x) - gamma1 * alpha * np.linalg.norm(gradient)**2:
            alpha *= delta

        if alpha < delta_k:
            return alpha
        
        # Goldstein conditions
        goldstein_condition_1 = self._problem.obj(x - alpha * gradient) < self._problem.obj(x) - gamma2 * alpha * np.linalg.norm(gradient)**2 
        goldstein_condition_2 = self._problem.obj(x - (alpha/delta) * gradient) < np.min([self._problem.obj(x - alpha * gradient), self._problem.obj(x) + gamma1 * (alpha/delta)*np.linalg.norm(gradient)**2])

        while goldstein_condition_1 and goldstein_condition_2:
            alpha /= delta
        
        return alpha
