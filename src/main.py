import pycutest
import tester as t

import matplotlib.pyplot as plt

def main():

    problem_instances = {
            #'AKIVA': pycutest.import_problem('AKIVA'),    
            #'ARGLBLE': pycutest.import_problem('ARGLBLE'),     
            'ARGLINA': pycutest.import_problem('ARGLINA'),    
            #'ARGLINC': pycutest.import_problem('ARGLINC'),    
            #'BOX': pycutest.import_problem('BOX'),              
            'BROYDN7D': pycutest.import_problem('BROYDN7D'),        
            #'CLIFF': pycutest.import_problem('CLIFF'), 
            #'DENSCHNA': pycutest.import_problem('DENSCHNA'),
            #'DJTL': pycutest.import_problem('DJTL'),
            #'BEALE': pycutest.import_problem('BEALE'),
               
        }
    
    tester = t.GradientDescentTester(problem_instances)
    test_results = tester.run_all_tests()

    tester.print_results_table(test_results)

if __name__ == "__main__":
    main()
