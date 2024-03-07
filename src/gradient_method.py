class GradientType:

    """
    Class that contains the gradient descent variants.

    Attributes
    ----------
    CONSTANT_STEPSIZE : int
        gradient descent with constant stepsize

    ARMIJO_LS : int
        gradient descent with Armijo line-search;

    ARMIJO_GOLDSTEIN : int
        gradient descent with Armijo line-search with expansion (Armijo-Goldstein)

    WOLFE_LS : int
        gradient descent with Wolfe line-search

    ARMIJO_NON_MONOTONE : int
        gradient descent with nonmonotone Armijo line-search

    BARZILAI_BORWEIN : int
        gradient descent with nonmonotone Armijo line-search, with initial stepsize
        set by the Barzilai-Borwein rule

    """

    CONSTANT_STEPSIZE = 1
    ARMIJO_LS = 2
    ARMIJO_GOLDSTEIN = 3
    WOLFE_LS = 4
    ARMIJO_NON_MONOTONE = 5
    BARZILAI_BORWEIN = 6

class GradientDescentMethod:

    """
    Class that contains the gradient descent method implementation with all the different variants specified by
    the type attribute.

    Attributes
    ----------
    type : GradientType
        type of gradient descend

    Methods
    -------

    """

    def __init__(self, method_type):
        self.type = method_type

    # metodi privati con __
    def __gradient_descent(self):
        pass