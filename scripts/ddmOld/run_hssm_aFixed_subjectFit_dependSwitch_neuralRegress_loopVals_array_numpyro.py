import os
import sys
import argparse
import hssm_aFixed_subjectFit_dependsSwitch_neuralRegress_bias_loopVals_numpyro
#note that we import hssm_aVals_func here

if __name__ == "__main__":
    
    #Add a parser
    parser = argparse.ArgumentParser()
    #Identity is the SBATCH number
    parser.add_argument('--identity', type = int)
    
    #Now that we have the arguments parsed, we decide what to do with them. 
    args = parser.parse_args()
    aVal = 3 + 0.5 * args.identity
    
    # if args.identity == 0:
    #     aVal = 3
    # elif args.identity == 1:
    #     aVal = 3.5
    # elif args.identity == 2:
    #     aVal = 4
    # elif args.identity == 3:
    #     aVal = 4.5
    # elif args.identity == 4:
    #     aVal = 5
    # elif args.identity == 5:
    #     aVal = 5.5
    # elif args.identity == 6:
    #     aVal = 6
    # elif args.identity == 7:
    #     aVal = 6.5
        
    
    #Now that we have the parsed argument, we can run things!
    my_saved_result = hssm_aFixed_subjectFit_dependsSwitch_neuralRegress_bias_loopVals_numpyro.hssm_aVals_func(aVal = aVal)
    
    #Then don't forget to save it
    '... pickle open block here'
    'pickle.dump(my_saved_result)'