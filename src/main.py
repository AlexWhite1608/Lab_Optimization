from gradient_method import GradientDescentMethod
import pycutest

import matplotlib.pyplot as plt

def main():

    # problem import
    problem = pycutest.import_problem('ROSENBR')
    #print(pycutest.problem_properties('ROSENBR'))

    # choose the method
    gradient_descend = GradientDescentMethod(problem=problem, max_iter=1000)

    # results
    x, iters = gradient_descend.gradient_descent_armijo_goldstein(delta_k=0.4, delta=0.1, gamma1=0.2, gamma2=0.4)
    #x, iters = gradient_descend.gradient_descent_armijo(delta_k=0.4, delta=0.3, gamma=0.3)

    print("Minimum @ %s after %s iterations" % (str(x), str(iters)))

if __name__ == "__main__":
    main()


#Minimum @ [0.99255198 0.98512658] after 1000 iterations
