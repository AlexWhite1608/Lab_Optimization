from time import time
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

        armijo_goldstein_params = {
            'delta_k': 0.4,
            'delta': 0.1,
            'gamma1': 0.2,
            'gamma2': 0.4
        }

        armijo_nm_params={
            'delta_k': 0.2,
            'delta': 0.1,
            'gamma': 0.1,
        }

        # Lista dei metodi di ottimizzazione da testare
        optimization_methods = [
            wolfe_ls.WolfeLS(gamma=0.2, sigma=0.3),
            constant_stepsize.ConstantStepSize(learning_rate=0.001),
            armijo_nm_ls.ArmijoNMLS(delta_k=0.4, delta=0.2, gamma=0.3),
            armijo_goldstein.ArmijoGoldstein(delta_k=0.4, delta=0.1, gamma1=0.2, gamma2=0.4),
            armijo_ls.ArmijoLS(delta_k=0.1, delta=0.2, gamma=0.4),
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
