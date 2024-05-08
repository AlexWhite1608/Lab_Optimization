import numpy as np
import matplotlib.pyplot as plt


# TODO: ELIMINA IL FILE!!

class GradientDescentMethod:
    """
    Classe che implementa i vari metodi specializzati del gradiente

    Variables:
        self._problem : Istanza del problema 
        self._x0 : Punto iniziale
        self._f : Funzione obiettivo
        self._tol (float): Tolleranza per l'ottimo
        self._max_iter (int): Numero massimo di iterazioni degli algoritmi
    """

    def __init__(self, problem, tol=1e-7, max_iter=1000):
        
        # Parametri del problema
        self._problem = problem
        self._x0 = problem.x0
        self._f = problem.obj(self._x0)

        # Settings
        self._tol = tol
        self._max_iter = max_iter

    def __init__(self, tol=1e-7, max_iter=1000):
        
        # Settings
        self._tol = tol
        self._max_iter = max_iter

    # Gradiente costante
    def gd_const(self, learning_rate=0.01):
        """
        Implementa il metodo del gradiente con tasso di apprendimento costante

        Args:
            learning_rate (float): Tasso di apprendimento costante

        Returns:
            x (ndarray): Punto di convergenza
            iterations (int): Numero di iterazioni eseguite
        """

        x = self._problem.x0
        for i in range(self._max_iter):
            gradient = self._problem.grad(x)
            if np.linalg.norm(gradient) < self._tol:
                break
            x = x - learning_rate * gradient

            print(f"Iteration {i+1}, x: {x}")

        return x, i + 1
    
    # Algoritmo di Armijo
    def gd_armijo(self, delta_k=0.5, delta=0.5, gamma=0.5):
        """
        Implementa il metodo del gradiente con ricerca di linea di Armijo

        Args:
            delta_k (float): Default: 0.5
            delta (float): Default: 0.5
            gamma (float): Default: 0.5

        Returns:
            x (ndarray): Punto di convergenza
            iterations (int): Numero di iterazioni eseguite
        """

        x = self._problem.x0
        for i in range(self._max_iter):
            gradient = self._problem.grad(x)
            if np.linalg.norm(gradient) < self._tol:
                break
            step_size = self.__armijo_ls(x, gradient, delta_k, delta, gamma)

            x = x - step_size * gradient

            print(f"Iteration {i+1}, (x,y): {x}")

        return x, i + 1
    
    # Algoritmo Armijo-Goldstein
    def gd_goldstein(self, delta_k=0.5, delta=0.5, gamma1=0.5, gamma2=0.5):
        """
        Implementa il metodo del gradiente con ricerca di linea di Goldstein

        Args:
            delta_k (float): Default: 0.5
            delta (float): Default: 0.5
            gamma1 (float): Default: 0.5
            gamma2 (float): Default: 0.5

        Returns:
            x (ndarray): Punto di convergenza.
            iterations (int): Numero di iterazioni eseguite
        """

        x = self._problem.x0
        for i in range(self._max_iter):
            gradient = self._problem.grad(x)
            if np.linalg.norm(gradient) < self._tol:
                break
            step_size = self.__armijo_goldstein_ls(x, gradient, delta_k, delta, gamma1, gamma2)

            x = x - step_size * gradient

            print(f"Iteration {i+1}, (x,y): {x}")

        return x, i + 1
    
    # Algoritmo LineSearch di Wolfe (ALGW2)
    def gd_wolfe(self, gamma, sigma):
        """
        Implementa il metodo del gradiente con ricerca di linea di Wolfe (ALGW2)

        Args:
            gamma (float): Parametro di controllo della condizione di Armijo
            sigma (float): Parametro di controllo della condizione di curvatura

        Returns:
            x (ndarray): Punto di convergenza
            iterations (int): Numero di iterazioni eseguite
        """

        x = self._problem.x0
        for i in range(self._max_iter):
            gradient = self._problem.grad(x)
            if np.linalg.norm(gradient) < self._tol:
                break
            step_size = self.__wolfe_ls(x, gradient, gamma, sigma)

            x = x - step_size * gradient

            print(f"Iteration {i+1}; (x,y): {x}")

        return x, i + 1
    
    # Algoritmo di Armijo non monotono
    def gd_armijo_non_monotone(self, delta_k=0.5, delta=0.5, gamma=0.5):
        """
        Implementa il metodo del gradiente con ricerca di linea di Armijo non monotona

        Args:
            delta_k (float): Parametro per il calcolo del valore iniziale W
            delta (float): Parametro per il calcolo del valore iniziale W
            gamma (float): Parametro di controllo per la condizione di Armijo non monotona

        Returns:
            x (ndarray): Punto di convergenza
            iterations (int): Numero di iterazioni eseguite
        """

        x = self._problem.x0
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
    
    # Algoritmo Armijo non monotono con formule BB (technique = {A, B, AB})
    def gd_barzilai_borwein(self, technique, armijo_goldstein_params={}, armijo_nm_params={}, sigma_a=1e-4, sigma_b=1e-4, M=15, N=10):
        """
        Implementa il metodo del gradiente con ricerca di linea di Barzilai-Borwein

        Args:
            technique (str): La tecnica per il calcolo di mu, può essere 'A','B' o 'AB'
            armijo_goldstein_params (dict): Parametri per la ricerca di linea di Goldstein
            armijo_nm_params (dict): Parametri per la ricerca di linea di Armijo non monotona
            sigma_a (float): Parametro per il test watchdog.
            sigma_b (float): Parametro per il test watchdog.
            M (int): Numero massimo di iterazioni esterne
            N (int): Numero massimo di iterazioni interne per la ricerca di linea di Barzilai-Borwein

        Returns:
            x (ndarray): Il punto di convergenza
            iterations (int): Il numero di iterazioni eseguite

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

    # Ricava mu per Armijo BB 
    def __get_mu(self, s, y, k, technique):
        """
        Calcola il valore di mu utilizzando le formule di Barzilai-Borwein a seconda della tecnica specificata

        Args:
            s (ndarray): La differenza tra i punti x_k e x_{k-1}
            y (ndarray): La differenza tra i gradienti di x_k e x_{k-1}
            k (int): Numero di iterazione corrente
            technique (str): La tecnica per il calcolo di mu, può essere 'A', 'B' o 'AB'

        Returns:
            float: Il valore di mu calcolato

        Raises:
            ValueError: Se la tecnica specificata non è valida
        """

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

    # Watchdog per Armijo BB
    def __watchdog_test(self, z_new, x, sigma_a, sigma_b, W):
        """
        Esegue il test watchdog per la ricerca di linea

        Args:
            z_new (ndarray): Il nuovo punto candidato ottenuto dalla ricerca di linea
            x (ndarray): Il punto corrente
            sigma_a (float): Parametro di controllo per il gradiente
            sigma_b (float): Parametro di controllo per la differenza tra punti
            W (float): Il valore massimo della funzione obiettivo nel punto corrente

        Returns:
            bool: True se il test è soddisfatto, False altrimenti
        """

        grad_norm = np.linalg.norm(self._problem.grad(z_new))
        diff_norm = np.linalg.norm(z_new - x)

        return self._problem.obj(z_new) <= W - max(sigma_a * grad_norm, sigma_b * diff_norm)

    # Armijo non monotono LineSearch
    def __armijo_nm(self, x, gradient, delta_k, delta, gamma, W):
        """
    Implementa la ricerca di linea di Armijo non monotona

    Args:
            x (ndarray): Il punto corrente
            gradient (ndarray): Il gradiente nel punto corrente
            delta_k (float): Passo iniziale per la ricerca di linea
            delta (float): Fattore di riduzione del passo
            gamma (float): Parametro di riduzione per la funzione obiettivo
            W (float): Il valore massimo della funzione obiettivo nel punto corrente

        Returns:
            float: Il passo alpha ottenuto dalla ricerca di linea
        """

        alpha = delta_k
        j=0

        while self._problem.obj(x - alpha * gradient) > W - gamma * alpha * np.dot(gradient, gradient):
            alpha *= delta
            j += 1

        return alpha

    # Calcola i massimi W per Armijo non monotono
    def __get_W(self, x, i, x_seq, M=3):
        """
        Calcola il valore di W

        Args:
            x (ndarray): Il punto corrente
            i (int): Indice dell'iterazione corrente
            x_seq (list): Lista dei punti visitati nelle iterazioni precedenti
            M (int, optional): Numero massimo di passi a ritroso per i valori precedenti

        Returns:
            float: Il valore del parametro W
        """

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
        """
        Implementa la ricerca di linea secondo le condizioni di Wolfe

        Args:
            x (ndarray): Il punto corrente
            gradient (ndarray): Il gradiente della funzione obiettivo nel punto corrente
            gamma (float): Parametro per la condizione di Wolfe
            sigma (float): Parametro per la condizione di Wolfe
            alpha_l (float, optional): Limite inferiore per la ricerca di linea. Default: 0
            alpha_u (float, optional): Limite superiore per la ricerca di linea. Default: 100

        Returns:
            float: Il passo di ricerca di linea che soddisfa le condizioni di Wolfe
        """

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
        """
        Implementa la ricerca di linea utilizzando la regola dell'Armijo

        Args:
            x (ndarray): Il punto corrente
            gradient (ndarray): Il gradiente della funzione obiettivo nel punto corrente
            delta_k (float): Passo iniziale per la ricerca di linea
            delta (float): Fattore di riduzione del passo
            gamma (float): Parametro per la regola dell'Armijo

        Returns:
            float: Il passo di ricerca di linea che soddisfa la regola dell'Armijo
        """

        alpha = delta_k
        while self._problem.obj(x - alpha * gradient) > self._problem.obj(x) - gamma * alpha * np.linalg.norm(gradient)**2:
            alpha *= delta

        return alpha
    
    def gd_armijo_goldstein(self, delta_k=0.4, delta=0.1, gamma1=0.2, gamma2=0.4):
        x = self._problem.x0
        for i in range(self._max_iter):
            gradient = self._problem.grad(x)
            if np.linalg.norm(gradient) < self._tol:
                break
            step_size = self.__armijo_goldstein_ls(x, gradient, delta_k, delta, gamma1, gamma2)

            x = x - step_size * gradient

            print(f"Iteration {i+1}, (x,y): {x}")

        return x, i + 1

    # Armijo-Goldstein LineSearch 
    def __armijo_goldstein_ls(self, x, gradient, delta_k, delta, gamma1, gamma2):
        """
        Implementa la ricerca di linea utilizzando la regola dell'Armijo con le condizioni di Goldstein

        Args:
            x (ndarray): Il punto corrente.
            gradient (ndarray): Il gradiente della funzione obiettivo nel punto corrente
            delta_k (float): Passo iniziale per la ricerca di linea
            delta (float): Fattore di riduzione del passo
            gamma1 (float): Parametro per la regola dell'Armijo
            gamma2 (float): Parametro per le condizioni di Goldstein

        Returns:
            float: Il passo di ricerca di linea che soddisfa le condizioni di Goldstein
        """

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
    
    def set_problem(self, problem):
        self._problem = problem
        return self
