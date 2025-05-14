import os
import matplotlib.pyplot as plt
import numpy as np

class GradientDescentMethod:
    def __init__(self, tol=1e-5, max_iter=1000):
        self._tol = tol
        self._max_iter = max_iter
        self._gradient_evaluations = 0
        self._objective_evaluations = 0
        self._x_history = []  
        self._obj_history = [] 
        self._name = None

    def optimize(self):
        pass

    def get_name(self):
        return self._name
    
    def set_problem(self, problem):
        self._problem = problem

    def evaluate_gradient(self, x):
        self._gradient_evaluations += 1
        return self._problem.grad(x)
    
    def evaluate_objective(self, arg):
        self._objective_evaluations += 1
        return self._problem.obj(arg)