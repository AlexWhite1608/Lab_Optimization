from time import time
import pycutest
from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np
import os
from methods_implementation import (
    armijo_goldstein,
    armijo_ls,
    armijo_nm_bb,
    armijo_nm_ls,
    constant_stepsize,
    wolfe_ls,
)

class GradientDescentTester:
    
    def __init__(self, problem_instances):
        self._problem_instances = problem_instances

    def run_all_tests(self):

        results = {}

        # Parametri della ricerca nm per armijo-BB
        armijo_nm_params={
            'delta_k': 0.3,
            'delta': 0.4,
            'gamma': 0.5,
        }

        #CLIFF Armijo NM: 0.2, 0.7, 0.8
        #CLIFF Armijo Costante: 0.0000000001

        optimization_methods = [
            wolfe_ls.WolfeLS(parameters={'gamma': 0.1, 'sigma': 0.4, 'alpha_l': 0, 'alpha_u': 100}),
            constant_stepsize.ConstantStepSize(parameters={'learning_rate': 0.01}),
            armijo_nm_ls.ArmijoNMLS(parameters={'delta_k': 0.1, 'delta': 0.4, 'gamma': 0.5}),
            armijo_goldstein.ArmijoGoldstein(parameters={'delta_k': 0.4, 'delta': 0.1, 'gamma1': 0.2, 'gamma2': 0.3}),
            armijo_ls.ArmijoLS(parameters={'delta_k': 0.1, 'delta': 0.2, 'gamma': 0.3}),
            armijo_nm_bb.ArmijoNMBB(technique='AB', armijo_nm_params=armijo_nm_params)
        ]

        # Esegue i test per ogni problema e ogni metodo 
        for problem_name, problem_instance in self._problem_instances.items():
            print(pycutest.problem_properties(problem_name))
            results[problem_name] = {}

            for method in optimization_methods:
                method.set_problem(problem_instance)
                problem_instance.x0 = problem_instance.x0.copy()
                
                start_time = time()
                solution, iterations = method.optimize()
                end_time = time()
                execution_time = end_time - start_time
                
                results[problem_name][method.get_name()] = {
                    'solution': solution,
                    'iterations': iterations,
                    'execution_time': execution_time,
                    'gradient_evaluations': method._gradient_evaluations,
                    'objective_evaluations': method._objective_evaluations,
                    'obj_history': method._obj_history  

                }

        return results
    
    def print_results_table(self, results):
        for problem_name, problem_results in results.items():
            method_names = list(problem_results.keys())
            table_data = []

            for method_name in method_names:
                method_results = [method_name]
                solution = problem_results[method_name]['solution']
                iterations = problem_results[method_name]['iterations']
                execution_time = problem_results[method_name]['execution_time']
                gradient_evaluations = problem_results[method_name]['gradient_evaluations']
                objective_evaluations = problem_results[method_name]['objective_evaluations']

                method_results.extend([solution, iterations, execution_time, gradient_evaluations, objective_evaluations])
                table_data.append(method_results)

            print(f"Results for problem: {problem_name} (n variables: {pycutest.problem_properties(problem_name)['n']})")
            print(tabulate(table_data, headers=['Method', 'Solution', 'Iterations', 'Execution Time', 'Gradient Evals', 'Objective Evals'], tablefmt="simple_grid"))
            print("\n")
    
    def plot_results(self, results):
        for problem_name, problem_results in results.items():
            plt.figure(figsize=(12, 8))
            
            # Trova la migliore soluzione globale e il metodo che l'ha trovata
            global_min_obj = float('inf')
            best_method = None
            
            for method_name, method_results in problem_results.items():
                obj_history = method_results.get('obj_history', [])
                print(f'obj history {method_name}: {obj_history[0:2]}')
                if obj_history:
                    min_obj = min(obj_history)
                    if min_obj < global_min_obj:
                        global_min_obj = min_obj
                        best_method = method_name
            
            for method_name, method_results in problem_results.items():
                obj_history = method_results.get('obj_history', [])
                if not obj_history:  
                    print(f"Warning: obj_history is empty for method {method_name} on problem {problem_name}")
                    continue
                
                scarti = []
                for i in range(len(obj_history)):
                    scarti.append(obj_history[i] - global_min_obj)
                print(f'scarti metodo {method_name}: {scarti[0:2]}')
                plt.plot(scarti, label=method_name)

            #plt.title(f'Convergence of Methods for Problem: {problem_name} (# variables: {pycutest.problem_properties(problem_name)["n"]})\n'
            #          f'Best solution found by: {best_method} with value: {global_min_obj}')
            plt.title(f'Convergence of Methods for Problem: {problem_name} (# variables: {pycutest.problem_properties(problem_name)["n"]})')
            plt.xlabel('Iteration')
            plt.ylabel('Difference from Best Objective Value')
            plt.yscale('log')
            plt.xscale('linear')  
            plt.xlim(-2, max(len(obj_history) for method_results in problem_results.values())) 
            plt.legend()
            plt.grid(True)

            base_folder = "plots"
            problem_folder = os.path.join(base_folder, problem_name)
            if not os.path.exists(problem_folder):
                os.makedirs(problem_folder)

            file_path = os.path.join(problem_folder, f"{problem_name}_convergence.png")
            plt.savefig(file_path)
            plt.close()
