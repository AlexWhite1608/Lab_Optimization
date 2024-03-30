from gradient_method import GradientDescentMethod
import pycutest

import matplotlib.pyplot as plt

def main():

    # problem import
    problem = pycutest.import_problem('CHNRSNBM')

    # choose the method
    gradient_descend = GradientDescentMethod(problem=problem, max_iter=1000)

    # results
    x, iters = gradient_descend.gradient_descent_armijo(alpha=0.5, delta=0.3, gamma=0.5, plotting=False)

    print("Minimum @ %s after %s iterations" % (str(x), str(iters)))

if __name__ == "__main__":
    main()
