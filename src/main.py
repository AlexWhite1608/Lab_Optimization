import pycutest
import tester as t

import matplotlib.pyplot as plt

def main():

    problem_instances = {
            #'BARD': pycutest.import_problem('BARD'),    
            'BRKMCC': pycutest.import_problem('BRKMCC'),     
            #'ARGLINA': pycutest.import_problem('ARGLINA'),    
            #'DIXMAANB': pycutest.import_problem('DIXMAANB'),    
            #'BOX': pycutest.import_problem('BOX'),              
            #'BROYDN7D': pycutest.import_problem('BROYDN7D'),        
            #'CLIFF': pycutest.import_problem('CLIFF'), 
            #'HIMMELBCLS': pycutest.import_problem('HIMMELBCLS'),
            #'HAIRY': pycutest.import_problem('HAIRY'),
            #'BEALE': pycutest.import_problem('BEALE')
        }
    
    
    tester = t.GradientDescentTester(problem_instances)
    test_results = tester.run_all_tests()

    tester.print_results_table(test_results)
    tester.plot_results(test_results)

if __name__ == "__main__":
    main()
