class GradientDescentMethod:
    def __init__(self, problem, tol=1e-7, max_iter=50):
        self._problem = problem
        self._tol = tol
        self._max_iter = max_iter
        self._name = None

    def __init__(self, tol=1e-7, max_iter=1000):
        self._tol = tol
        self._max_iter = max_iter
        self._name = None

    def optimize(self):
        pass

    def get_name(self):
        return self._name
    
    def set_problem(self, problem):
        self._problem = problem
        return self