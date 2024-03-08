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

    """
    Class that contains the gradient descent method implementation with all the different variants specified by
    the type attribute.

    Attributes
    ----------
    type : GradientType
        type of gradient descend

    Methods
    -------

    """

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
        self.problem = problem
        self.x0 = problem.x0
        self.f = problem.obj(self.x0)

        # convergence settings
        self.tol = tol
        self.max_iter = max_iter
        pass

    def gradient_descent_const(self, learning_rate=0.01, plotting=True):
        x = self.x0
        for i in range(self.max_iter):
            gradient = self.problem.grad(x)
            if np.linalg.norm(gradient) < self.tol:
                break
            x = x - learning_rate * gradient

            print(f"Iteration {i+1}, x: {x}")

            # plotting
            if plotting:
                #TODO: plotting dinamico
                pass

        return x, i + 1
    
    def __plot(self, x, y, current_pos):
        plt.plot(x, y)
        plt.scatter(current_pos[0], current_pos[1], color='red')
        plt.pause(0.001)
        plt.clf()

