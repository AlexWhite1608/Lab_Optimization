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
        self.problem = problem
        self.x0 = problem.x0
        self.f = problem.obj(self.x0)

        # convergence settings
        self.tol = tol
        self.max_iter = max_iter
        pass

    # GRADIENTE COSTANTE
    def gradient_descent_const(self, learning_rate=0.01, plotting=True):
        x = self.x0
        for i in range(self.max_iter):
            gradient = self.problem.grad(x)
            if np.linalg.norm(gradient) < self.tol:
                break
            x = x - learning_rate * gradient

            print(f"Iteration {i+1}, x: {x}")

            #TODO: plotting dinamico
            if plotting:
                pass

        return x, i + 1
    
    # LS ARMIJO
    def gradient_descent_armijo(self, alpha=0.5, delta=0.5, gamma=0.5, plotting=False):
        x = self.x0
        for i in range(self.max_iter):
            gradient = self.problem.grad(x)
            if np.linalg.norm(gradient) < self.tol:
                break
            step_size = self.__armijo_ls(x, gradient, alpha, delta, gamma)

            current_pos = x
            x = x - step_size * gradient

            print(f"Iteration {i+1}, (x,y): {x}")

            #TODO: plotting dinamico
            if plotting:
                self.__plot(x=x[0], y=x[1], current_pos=current_pos)

        return x, i + 1

    # Armijo LineSearch 
    def __armijo_ls(self, x, gradient, alpha, delta, gamma):
        
        while self.problem.obj(x - alpha * gradient) > self.problem.obj(x) - gamma * alpha * np.linalg.norm(gradient)**2:
            alpha *= delta
        return alpha

    # Plotting method
    def __plot(self, x, y, current_pos):
        # Genera un meshgrid per visualizzare la funzione obiettivo
        x_range = np.linspace(np.min(x) - 1, np.max(x) + 1, 100)
        y_range = np.linspace(np.min(y) - 1, np.max(y) + 1, 100)
        X, Y = np.meshgrid(x_range, y_range)
        Z = self.problem.obj(np.array([X, Y]))  # Converti [X, Y] in un array numpy

        # Plot della superficie della funzione obiettivo
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis')

        # Plot del percorso della ricerca del minimo
        ax.scatter(x, y, self.problem.obj([x, y]), color='red', s=50, label='Current Position')
        ax.scatter(current_pos[0], current_pos[1], self.problem.obj(current_pos), color='blue', s=50, label='Next Position')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Objective Function Value')
        ax.legend()

        plt.show()

