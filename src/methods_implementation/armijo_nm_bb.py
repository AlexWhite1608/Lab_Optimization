from . import gradient
import numpy as np

class ArmijoNMBB(gradient.GradientDescentMethod):

    def __init__(self,  technique, armijo_goldstein_params={}, armijo_nm_params={}, sigma_a=1e-4, sigma_b=1e-4, M=15, N=10, max_iter=1000):
        super().__init__(max_iter=max_iter)
        self._technique = technique
        self._armijo_goldstein_params = armijo_goldstein_params
        self._armijo_nm_params = armijo_nm_params
        self._sigma_a = sigma_a
        self._sigma_b = sigma_b
        self._M = M
        self._N = N
        self._name = 'Barzilai-Borwein Line Search'

    def optimize(self):        
        """
        Note:
            In base al valore passato al parametro technique si va ad assegnare un valore diverso al parametro mu.
            Se technique='AB' allora per le iterazioni pari si utilizza la tecnica A, e per quelle dispari la tecnica B
        """

        k = 0
        x = self._problem.x0
        x_seq = []

        # Inizializza x_k_minus_1 e grad_k_minus_1
        x_k_minus_1 = x
        grad_k_minus_1 = self._problem.grad(x)

        # Parametri per ricerca Goldstein per ricavare mu se è nullo
        delta_k_gs = self._armijo_goldstein_params['delta_k']
        delta_gs = self._armijo_goldstein_params['delta']
        gamma1_gs = self._armijo_goldstein_params['gamma1']
        gamma2_gs = self._armijo_goldstein_params['gamma2']

        # Parametri per ricerca Armijo nm per ricavare alpha
        delta_k_nm = self._armijo_nm_params['delta_k']
        delta_nm = self._armijo_nm_params['delta']
        gamma_nm = self._armijo_nm_params['gamma']

        while True:
            gradient = self._problem.grad(x)
            norm_grad = np.linalg.norm(gradient)

            if norm_grad < self._tol or k >= self._max_iter:
                break

            z_k = x
            linesearch = True

            for i in range(self._N):

                # Calcola i valori di s e y
                s = z_k - x_k_minus_1
                grad_k = self._problem.grad(z_k)
                y = grad_k - grad_k_minus_1

                # Calcola mu utilizzando i valori di s e y e la tecnica fornita in input
                mu = self.__get_mu(s, y, k)  

                if mu == 0:
                    mu = self.__armijo_goldstein_ls(delta_k_gs, delta_gs, gamma1_gs, gamma2_gs)  # Ricava mu con Goldstein 

                # Calcola z_new utilizzando il valore di mu
                z_new = z_k - (1 / mu) * self._problem.grad(z_k)

                # Esegue il test watchdog
                if self.__watchdog_test(z_new, x, self.__get_W(x, k, x_seq)):
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

            print(f"{self._name}: {k}; (x,y): {x}")

        print("----------------------------------------------\n")
        return x, k
    
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
    
    def __get_mu(self, s, y, k):
        # TODO: verifica che mu stia dentro i valori imposti da epsilon=1e-10

        if self._technique == 'A':
            return np.dot(s, y) / np.dot(s, s)
        elif self._technique == 'B':
            return np.dot(y, y) / np.dot(s, y)
        elif self._technique == 'AB':

            # Se k (iterazione) è pari allora faccio A, altrimenti B
            if k % 2 == 0:
                return np.dot(s, y) / np.dot(s, s)
            else:
                return np.dot(y, y) / np.dot(s, y)
        else:
            raise ValueError("Invalid mu technique")
        
    def __watchdog_test(self, z_new, x, W):
        grad_norm = np.linalg.norm(self._problem.grad(z_new))
        diff_norm = np.linalg.norm(z_new - x)

        return self._problem.obj(z_new) <= W - max(self._sigma_a * grad_norm, self._sigma_b * diff_norm)

    def __armijo_nm(self, x, gradient, delta_k, delta, gamma, W):
        alpha = delta_k
        j=0

        while self._problem.obj(x - alpha * gradient) > W - gamma * alpha * np.dot(gradient, gradient):
            alpha *= delta
            j += 1

        return alpha
    
    def __armijo_goldstein_ls(self, x, gradient, delta_k, delta, gamma1, gamma2):
        alpha = delta_k
        while self._problem.obj(x - alpha * gradient) > self._problem.obj(x) - gamma1 * alpha * np.linalg.norm(gradient)**2:
            alpha *= delta

        if alpha < delta_k:
            return alpha
        
        # Condizioni di Goldstein
        goldstein_condition_1 = self._problem.obj(x - alpha * gradient) < self._problem.obj(x) - gamma2 * alpha * np.linalg.norm(gradient)**2 
        goldstein_condition_2 = self._problem.obj(x - (alpha/delta) * gradient) < np.min([self._problem.obj(x - alpha * gradient), self._problem.obj(x) + gamma1 * (alpha/delta)*np.linalg.norm(gradient)**2])

        while goldstein_condition_1 and goldstein_condition_2:
            alpha /= delta
        
        return alpha