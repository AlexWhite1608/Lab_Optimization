from gradient_method import GradientDescentMethod
import pycutest

import matplotlib.pyplot as plt

#TODO: usa tabulate!!

def main():

    # problem import
    problem = pycutest.import_problem('ROSENBR')
    #print(pycutest.problem_properties('ROSENBR'))

    # choose the method
    gradient_descend = GradientDescentMethod(problem=problem, max_iter=1000)

    #TODO: parametri estratti con numpy.random.uniform()

    # results
    #x, iters = gradient_descend.gd_armijo_goldstein(delta_k=0.4, delta=0.1, gamma1=0.2, gamma2=0.4)
    #x, iters = gradient_descend.gd_armijo(delta_k=0.4, delta=0.3, gamma=0.3)
    #x, iters = gradient_descend.gd_wolfe(gamma=0.2, sigma=0.3)
    #x, iters = gradient_descend.gd_armijo_non_monotone(delta_k=0.4, delta=0.2, gamma=0.3)

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
    x, iters = gradient_descend.gd_barzilai_borwein(technique='B', armijo_goldstein_params=armijo_goldstein_params, armijo_nm_params=armijo_nm_params)

    print("Minimum @ %s after %s iterations" % (str(x), str(iters)))

if __name__ == "__main__":
    main()
