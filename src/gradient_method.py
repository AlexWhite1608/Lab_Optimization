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

    def __init__(self, problem, tol=1e-7, max_iter=1000):
        
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
            step_size = self.__wolfe_ls(x, gradient, gamma, sigma, i)

            x = x - step_size * gradient

            print(f"Iteration {i+1}; (x,y): {x}")

        return x, i + 1
    
    # ALGORITMO LS ARMIJO NON MONOTONE
    def gd_armijo_non_monotone(self, delta_k=0.5, delta=0.5, gamma=0.5):
        x = self._x0
        x_seq = [] 

        for i in range(self._max_iter):
            gradient = self._problem.grad(x)
            if np.linalg.norm(gradient) < self._tol:
                break
            W = self.__get_W(x, i, x_seq)
            step_size = self.__armijo_nm(x, gradient, delta_k, delta, gamma, W)

            x = x - step_size * gradient

            print(f"Iteration {i+1}; (x,y): {x}")

        return x, i + 1
    
    # ALGORITMO LS BARZILAI BORWEIN (technique = {A, B, AB})
    def gd_barzilai_borwein(self, technique, armijo_goldstein_params={}, armijo_nm_params={}, sigma_a=1e-4, sigma_b=1e-4, M=15, N=10):
        k = 0
        x = self._x0
        x_seq = []

        # Inizializza x_k_minus_1 e grad_k_minus_1
        x_k_minus_1 = x
        grad_k_minus_1 = self._problem.grad(x)

        # Parametri per ricerca Goldstein per ricavare mu se è nullo
        delta_k_gs = armijo_goldstein_params['delta_k']
        delta_gs = armijo_goldstein_params['delta']
        gamma1_gs = armijo_goldstein_params['gamma1']
        gamma2_gs = armijo_goldstein_params['gamma2']

        # Parametri per ricerca Armijo nm per ricavare alpha
        delta_k_nm = armijo_nm_params['delta_k']
        delta_nm = armijo_nm_params['delta']
        gamma_nm = armijo_nm_params['gamma']

        while True:
            gradient = self._problem.grad(x)
            norm_grad = np.linalg.norm(gradient)

            if norm_grad < self._tol or k >= self._max_iter:
                break

            z_k = x
            linesearch = True

            for i in range(N):

                # Calcola i valori di s e y
                s = z_k - x_k_minus_1
                grad_k = self._problem.grad(z_k)
                y = grad_k - grad_k_minus_1

                # Calcola mu utilizzando i valori di s e y e la tecnica fornita in input
                mu = self.__get_mu(s, y, k, technique)  

                if mu == 0:
                    mu = self.__armijo_goldstein_ls(delta_k_gs, delta_gs, gamma1_gs, gamma2_gs)  # Ricava mu con Goldstein 

                # Calcola z_new utilizzando il valore di mu
                z_new = z_k - (1 / mu) * self._problem.grad(z_k)

                # Esegue il test watchdog
                if self.__watchdog_test(z_new, x, sigma_a, sigma_b, self.__get_W(x, k, x_seq)):
                    x = z_new
                    linesearch = False
                    break

            if linesearch:
                alpha = self.__armijo_nm(x, gradient, delta_k_nm, delta_nm, gamma_nm, self.__get_W(x, k, x_seq))
                x = x - alpha * gradient

            # Aggiorna x_k_minus_1 e grad_k_minus_1 per la prossima iterazione
            x_k_minus_1 = z_k
            grad_k_minus_1 = gradient

            k += 1

            print(f"Iteration {k}; (x,y): {x}")

        return x, k

    
    # Ricava mu per Armijo_BB 
    def __get_mu(self, s, y, k, technique):

        # TODO: verifica che mu stia dentro i valori imposti da epsilon=1e-10

        if technique == 'A':
            return np.dot(s, y) / np.dot(s, s)
        elif technique == 'B':
            return np.dot(y, y) / np.dot(s, y)
        elif technique == 'AB':
            # Se k (iterazione) è pari allora faccio A, altrimenti B
            if k % 2 == 0:
                return np.dot(s, y) / np.dot(s, s)
            else:
                return np.dot(y, y) / np.dot(s, y)
        else:
            raise ValueError("Invalid mu technique")

    # Watchdog per Armijo_BB
    def __watchdog_test(self, z_new, x, sigma_a, sigma_b, W):
        grad_norm = np.linalg.norm(self._problem.grad(z_new))
        diff_norm = np.linalg.norm(z_new - x)

        return self._problem.obj(z_new) <= W - max(sigma_a * grad_norm, sigma_b * diff_norm)

    # Armijo Non Monotone LineSearch
    def __armijo_nm(self, x, gradient, delta_k, delta, gamma, W):
        alpha = delta_k
        j=0

        while self._problem.obj(x - alpha * gradient) > W - gamma * alpha * np.dot(gradient, gradient):
            alpha *= delta
            j += 1

        return alpha

    # Calcola la sequenza dei massimi W per Armijo-NM
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
