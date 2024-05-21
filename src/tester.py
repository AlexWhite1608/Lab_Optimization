from time import time
from tabulate import tabulate
import matplotlib.pyplot as plt
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

        # Parametri della ricerca goldstein per armijo-BB
        armijo_goldstein_params = {
            'delta_k': 0.3,
            'delta': 0.1,
            'gamma1': 0.2,
            'gamma2': 0.3
        }

        # Parametri della ricerca nm per armijo-BB
        armijo_nm_params={
            'delta_k': 0.3,
            'delta': 0.1,
            'gamma': 0.3,
        }

        optimization_methods = [
            wolfe_ls.WolfeLS(parameters={'gamma': 0.2, 'sigma': 0.4, 'alpha_l': 0, 'alpha_u': 100}),
            constant_stepsize.ConstantStepSize(parameters={'learning_rate': 0.01}),
            armijo_nm_ls.ArmijoNMLS(parameters={'delta_k': 0.5, 'delta': 0.2, 'gamma': 0.3}),
            armijo_goldstein.ArmijoGoldstein(parameters={'delta_k': 0.4, 'delta': 0.1, 'gamma1': 0.2, 'gamma2': 0.3}),
            armijo_ls.ArmijoLS(parameters={'delta_k': 0.1, 'delta': 0.2, 'gamma': 0.3}),
            armijo_nm_bb.ArmijoNMBB(technique='AB', armijo_goldstein_params=armijo_goldstein_params, armijo_nm_params=armijo_nm_params)
        ]

        # Esegue i test per ogni problema e ogni metodo 
        for problem_name, problem_instance in self._problem_instances.items():
            results[problem_name] = {}

            for method in optimization_methods:
                method.set_problem(problem_instance)
                
                start_time = time()
                solution, iterations = method.optimize()
                end_time = time()
                execution_time = end_time - start_time
                
                results[problem_name][method.get_name()] = {
                    'solution': solution,
                    'iterations': iterations,
                    'execution_time': execution_time
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

                method_results.extend([solution, iterations, execution_time])
                table_data.append(method_results)

            print(f"Results for problem: {problem_name}")
            print(tabulate(table_data, headers=['Method', 'Solution', 'Iterations', 'Execution Time'], tablefmt="simple_grid"))
            print("\n")