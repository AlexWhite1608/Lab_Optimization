from gradient_method import GradientDescentMethod
import pycutest

def main():

    problem = pycutest.import_problem('ROSENBR')
    gradient_descend = GradientDescentMethod(problem=problem, max_iter=100)

    x, iters = gradient_descend.gradient_descent_const(learning_rate=0.01)

    print("Found minimum x = %s after %g iterations" % (str(x), iters))

if __name__ == "__main__":
    main()
