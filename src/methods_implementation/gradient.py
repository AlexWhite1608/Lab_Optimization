import os
import matplotlib.pyplot as plt
import numpy as np

class GradientDescentMethod:
    def __init__(self, tol=1e-7, max_iter=500):
        self._tol = tol
        self._max_iter = max_iter
        self._x_history = []  
        self._obj_history = [] 
        self._name = None

    def optimize(self):
        pass

    def get_name(self):
        return self._name
    
    def set_problem(self, problem):
        self._problem = problem

    def plot_results(self):

        x_vals = np.array(self._x_history)
        obj_vals = np.array(self._obj_history)

        min_index = np.argmin(obj_vals)
        min_x = x_vals[min_index]
        min_obj = obj_vals[min_index]

        plt.figure(figsize=(10, 6))

        plt.plot(obj_vals, label='Objective function')
        plt.scatter(min_index, min_obj, color='red', zorder=5, label='Minimum')
        plt.title(f"{self._name}: {self._problem.name}")
        plt.xlabel('Iteration')
        plt.ylabel('Objective function value')
        plt.legend()
        plt.grid(True)

        # Annotazione del punto di minimo
        plt.annotate(f'Minimum: {min_obj:.4f}', xy=(min_index, min_obj), xytext=(min_index, min_obj + 0.05 * (max(obj_vals) - min(obj_vals))))

        base_folder = "plots"
        problem_folder = os.path.join(base_folder, self._problem.name)
        if not os.path.exists(problem_folder):
            os.makedirs(problem_folder)

        file_path = os.path.join(problem_folder, f"{self._name}.png")
        plt.savefig(file_path)
        plt.close()