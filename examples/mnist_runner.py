import numpy as np
import os
import sys
import h5py 

mydir = os.path.dirname(__file__)
sys.path.append(os.path.join(mydir, '..'))

import dbn_runner
from common import num_mistakes,logger

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run dnb on mnist benchmark')

    parser.add_argument('--train_file', type=str,required=True)    

    args, unknown = parser.parse_known_args()
    
    print('Data files:')
    for k,v in args.__dict__.items():
        print('\t%s %s' % (k, v))
        
    f = h5py.File(args.train_file,'r')
    trainInps = f['train_data']
    testInps = f['test_data']
    trainTargs = f['train_labels'][...]
    testTargs = f['test_labels'][...]
    
    assert(trainInps.shape == (60000, 784))
    assert(trainTargs.shape == (60000, 10))
    assert(testInps.shape == (10000, 784))
    assert(testTargs.shape == (10000, 10))
    
    
    positionalargs=[trainInps,trainTargs,testInps,testTargs]
    
    for predicted,targets,net in dbn_runner.main(*positionalargs):    
        error = num_mistakes(targets,predicted)/float(targets.shape[0])
        print("\tTest error rate:%g" % error)

    f.close()