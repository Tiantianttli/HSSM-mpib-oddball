import os
import sys
import argparse
import lba_real_singleSub
#note that we import ddm_real_singleSub here

if __name__ == "__main__":
    
    #Add a parser
    parser = argparse.ArgumentParser()
    #Identity is the SBATCH number
    parser.add_argument('--identity', type = int)
    
    #Now that we have the arguments parsed, we decide what to do with them. 
    args = parser.parse_args()
    subNum = args.identity

    #Now that we have the parsed argument, we can run things!
    lba_real_singleSub.lba_real_singleSub(subNum = subNum)
    