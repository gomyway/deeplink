import numpy as np
import math
import itertools
import sys
import time
import os
import random
import h5py
import logging
import pandas as pd
import csv

mydir = os.path.dirname(__file__)
sys.path.append(os.path.join(mydir, '.'))

from cudamat import gnumpy as gp


class Unbuffered:
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)
   
   
sys.stdout = Unbuffered(sys.stdout)  # for logging into file to work 
    
# wait gpu
_waitGpu = os.environ.get('DEEPLINK_WAIT_GPU', 'no')

if(_waitGpu == 'yes'):    
    locked_gpu = False
    total_wait_seconds = 0
    while(not locked_gpu):
        try:
            gp.garray(np.zeros(1))
            locked_gpu = True
            print 'GPU board is available after waited %d seconds' % total_wait_seconds
        except:
            locked_gpu = False
            if(total_wait_seconds==0): print 'No GPU board is available, waiting...'
            seconds = 600 + random.randint(-500, 500)            
            time.sleep(seconds)  # sleep 5 minutes to re-check
            total_wait_seconds += seconds


# create logger
logger = logging.getLogger('deeplink_logger')
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)

def get_batches(data, batch_size):
    '''
    batch_size: an integer, size of each training batch
    '''
    m = data.shape[0]
    num_batches = int(math.ceil(1.0 * m / batch_size))
    bsize = batch_size #int(m / num_batches)
    
    batch_sizes = np.zeros(num_batches, 'i') 
    for i in range(num_batches):
        i1 = i * bsize
        i2 = min(m, i1 + bsize)
        batch_sizes[i] = i2 - i1
        #print "%d"% (batch_sizes[i])        
    # print 'There are '+str(num_batches)+' batches'
    return bsize, batch_sizes


def garrayify(arrays):
    return [ar if isinstance(ar, gp.garray) else gp.garray(ar) for ar in arrays]

def numpyify(arrays):
    return [ar if isinstance(ar, np.ndarray) else ar.as_numpy_array(dtype=np.float32) for ar in arrays]


def columnRMS(W):
    return gp.sqrt(gp.mean(W * W, axis=0))


def limitColumnRMS(W, rmsLim):
    """
    All columns of W with rms entry above the limit are scaled to equal the limit.
    The limit can either be a row vector or a scalar.
    """
    rmsScale = rmsLim / columnRMS(W)
    return W * (1 + (rmsScale < 1) * (rmsScale - 1))


def bernoulli(gpu_hidden):
    return gpu_hidden.rand() < gpu_hidden

def force_gpu_deallocate(a):
   if(not isinstance(a, gp.garray)): return
   
   try:
       thesize = a.shape[0] * a.shape[1]
       del a  # delete variable    
       gp._cmsForReuse[thesize].pop()
       gp.__memoryInUse -= thesize * 4
       del gc.garbage[:]
   except IndexError:
       # Can't deallocate if it don't exist, can ya?
       pass

class gpu_buffer(object):
    '''
    A gpu memory buffer manager, which makes efficient usage of gpu memory: 
        1. reuse previously allocated gpu buffer whenever it is possible
        2. free gpu memory as soon as it's not being used 
        2. reduce cpu <-> gpu IO as much as possible  
    '''
    _gpu_buffer = None  # static class level data member
    _gpu_label_buffer = None
    _gpu_buffer_size = 0  # number of float32
        
    def __init__(self, data, label=None, batch_size=128, randomize=False, gpu_buffer_size_MB=512):
        nsize = int(gpu_buffer_size_MB * 1024 * 1024 / 4)  # float32 take 4 bytes        
        
        self.data = data
        self.label = label
        self.batch_size = batch_size
        self.randomize = randomize
        
        columns = self.data.shape[1] if self.label is None else 1 + self.data.shape[1] 
        
        self.num_rows = int(nsize / columns)  # maximum number of rows can hold in buffer
        # align with the minibatch_size
        self.num_rows = int(self.num_rows / self.batch_size) * self.batch_size
        #print "number of rows to be allocated in gpu: %d"%self.num_rows
        # allocate gpu memory only once, reuse after worth if possible
        
        if(gpu_buffer._gpu_buffer_size < self.num_rows * self.data.shape[1]):
            # requesting bigger buffer, has to re-allocate and resize        
            print "requesting buffer with size of %d float32" % (self.num_rows * self.data.shape[1])    
            # if(gpu_buffer._gpu_buffer is not None): gpu_buffer._gpu_buffer._free_device_memory()
            gpu_buffer._gpu_buffer = gp.zeros((self.num_rows, self.data.shape[1]))                                    
            gpu_buffer._gpu_buffer_size = self.num_rows * self.data.shape[1]
                    
        # if(gpu_buffer._gpu_label_buffer is not None): gpu_buffer._gpu_label_buffer._free_device_memory() 
        if self.label is not None:           
            gpu_buffer._gpu_label_buffer = gp.zeros((self.num_rows, self.label.shape[1])) 
            
    def _gdata(self, data):
        return gpu_buffer._gpu_buffer._overwrite(data)
        
    def _glabel(self, label):        
        # return gp.garray(label) if label is not None else None
        return gpu_buffer._gpu_label_buffer._overwrite(label) if label is not None else None
           
    def _iter_random_buffer(self):
        while True:
            idx = np.random.randint(self.data.shape[0], size=(self.num_rows,))
            gp_label_buffer = None if self.label is None else self._glabel(self.label[idx])            
            yield self._gdata(self.data[idx]), gp_label_buffer
    
    def _iter_all_buffer(self):        
        num_buffers = int(self.data.shape[0] / self.num_rows) + 1  # +1 to handle possible remainder that smaller then buffer_size
        (start, end) = (0, 0)
        for i in range(num_buffers):
            (start, end) = (i * self.num_rows, (i + 1) * self.num_rows) 
            if(end > self.data.shape[0]): end = self.data.shape[0]           
            if(start < self.data.shape[0]):
                gp_label_buffer = None if self.label is None else self._glabel(self.label[start:end])
                yield self._gdata(self.data[start:end]), gp_label_buffer
            
    def iter_minibatch(self):
        buffer_iter = self._iter_random_buffer() if self.randomize else self._iter_all_buffer()        
        for data_buffer, label_buffer in buffer_iter:  
            num_batches_per_buffer = int(data_buffer.shape[0] / self.batch_size) + 1  # +1 to handler overflow
            for i in range(num_batches_per_buffer):
                (start, end) = (i * self.batch_size, (i + 1) * self.batch_size)
                if(end > data_buffer.shape[0]): end = data_buffer.shape[0]
                if(start < data_buffer.shape[0]):
                    gp_label_batch = None if label_buffer is None else label_buffer[start : end]
                    yield data_buffer[start : end], gp_label_batch


class cpu_buffer(object):
    ''' A cpu memory buffer manager, which makes possible to use data set large than cpu memory: 
        1. use hdf5 file to reduce disk I/O as much as possible
        2. make efficent cpu memory usage and fast disk i/o with hdf5 data file
        3. works perfectly with SSD disk (e.g. 256GB), as replacement for expensive RAM  
    '''    
    def __init__(self, dataset, label=None, cpu_buffer_size_MB=4096):
        '''dataset is a h5py data_set, resident in disk and support slicing into numpy array (resident in cpu memory)'''
        self.dataset = dataset
        self.label = label
        nsize = int(cpu_buffer_size_MB * 1024 * 1024 / 4)  # float32 take 4 bytes    
        self.num_rows = int(nsize / self.dataset.shape[1])  # maximum number of rows can hold in mem
        if self.dataset.shape[0] <= self.num_rows: self.dataset = self.dataset[...]  # load whole dataset to memory if dataset is small enough
        self.num_buffers = int(self.dataset.shape[0] / self.num_rows) + 1  # +1 to handle possible remainder that smaller then buffer_size
        
    def iter_random_mem(self):
        mem_size = min(self.num_rows, self.dataset.shape[0])
        while True:            
            idx = np.random.randint(self.dataset.shape[0], size=(mem_size,))
            u, indices = np.unique(idx, return_inverse=True)
            label = None if self.label is None else self.label[idx]
            sys.stdout.write(".")
            yield self.dataset[u, :][indices], label  # h5py data_set can only be sliced this way for sampling with replacement

    def iter_all_mem(self):        
        (start, end) = (0, 0)        
        for i in range(self.num_buffers):
            (start, end) = (i * self.num_rows, (i + 1) * self.num_rows) 
            if(end > self.dataset.shape[0]): end = self.dataset.shape[0]           
            if(start < self.dataset.shape[0]):
                label = None if self.label is None else self.label[start:end]
                sys.stdout.write(".")
                yield self.dataset[start:end], label
    
class cpu_gpu_buffer(cpu_buffer):
    '''
    A cpu gpu memory buffer manager, which makes efficient usage of both cpu and gpu memory
    '''
    def __init__(self, data, label=None, batch_size=128, randomize=False, gpu_buffer_size_MB=512, cpu_buffer_size_MB=4096, inverse_disk_io_rate=1):
        '''inverse_disk_io_rate>1, controls how less frequently we want to do disk io, with the cost of reusing smaller data in mem, 
            inverse_disk_io_rate = 1 means we always want fresh data once all data in mem has been consumed'''
        if(not isinstance(data, h5py.Dataset)):
            raise ValueError("Input data is not of h5py.Dataset type")
        
        num_gpu_buffers = int(cpu_buffer_size_MB / gpu_buffer_size_MB)
        
        cpu_buffer_size_MB = num_gpu_buffers * gpu_buffer_size_MB  # enforce that cpu_buffer_size is divisible by gpu_buffer_size_MB
        
        super(cpu_gpu_buffer, self).__init__(data, label, cpu_buffer_size_MB)
        
        self.cpu_mem_iter = self.iter_random_mem() if randomize else self.iter_all_mem()
        
        self.batch_size = batch_size
        self.randomize = randomize
        self.gpu_buffer_size_MB = gpu_buffer_size_MB
        
        nsize = int(gpu_buffer_size_MB * 1024 * 1024 / 4)  # float32 take 4 bytes 
        num_gpu_rows = int(nsize / data.shape[1])
        
        self.num_batches = inverse_disk_io_rate * num_gpu_buffers * int(num_gpu_rows / batch_size)
            
    def iter_minibatch(self):
        for cpu_mem, cpu_label in self.cpu_mem_iter:
            minibatch = gpu_buffer(cpu_mem, cpu_label, self.batch_size, self.randomize, self.gpu_buffer_size_MB).iter_minibatch()
            for n, (data, label) in enumerate(minibatch):
                yield data, label
                if(n > self.num_batches): break  # time to load a new piece of cpu_mem from disk

# may use this for prediction tasks, which is fast on cpu anyway
def iter_minibatch(data, batch_size): 
    bsize, batch_sizes = get_batches(data, batch_size) 
    num_batches = batch_sizes.shape[0]
    
    for i in range(num_batches):
        (start, end) = (i * bsize, i * bsize + batch_sizes[i])
        yield data[start : end], (start, end)

def iter_minibatch_tuple(data, labels, batch_size): 
    bsize, batch_sizes = get_batches(data, batch_size) 
    num_batches = batch_sizes.shape[0]                 
    for i in range(num_batches):
        (start, end) = (i * bsize, i * bsize + batch_sizes[i])
        yield data[start : end], labels[start : end], (start, end)

def sampleMinibatch(mbsz, inps, targs):
    idx = np.random.randint(inps.shape[0], size=(mbsz,))
    return inps[idx], targs[idx]


'''
Two of the most useful ways to standardize inputs are: 

 o Mean 0 and standard deviation 1 
 o Midrange 0 and range 2 (i.e., minimum -1 and maximum 1) 
 
 http://www.faqs.org/faqs/ai-faq/neural-nets/part2/
 
'''
def normalize(data, mu=None, sigma=None):
    '''[0,1]
    data normalization function
    data : 2D array, each row is one data
    mu   : 1D array, each element is the mean the corresponding column in data
    sigma: 1D array, each element is the standard deviation of the corresponding
           column in data
    '''
    (m, n) = data.shape
    if mu is None or sigma is None:
        mu = np.mean(data, 0)
        sigma = np.std(data, 0)

    mu_rep = np.tile(mu, (m, 1))
    sigma_rep = np.tile(sigma, (m, 1))
    return (data - mu_rep) / sigma_rep, mu, sigma

def un_normalize(data, mu, sigma):
    '''
    un-normalize the normalized data. This is used for visualization purpose
       data : 2D array, each row is one data
    mu   : 1D array, each element is the mean the corresponding column in data
    sigma: 1D array, each element is the standard deviation of the corresponding
           column in data
    '''
    (m, n) = data.shape
    mu_rep = np.tile(mu, (m, 1))
    sigma_rep = np.tile(sigma, (m, 1))
    return np.multiply(data, sigma_rep) + mu_rep

'''
    standardize the target variables, usually use normalize_minus_one or normalize_zero_one as below
'''

def normalize_zero_one(data, min=None, gap=None):
    ''' 
    normalize data to range [0,1], also for normalizing targets if activation function is sigmoid()
    data : 2D array, each row is one data
    min   : 1D array, each element is the min of the corresponding column in data
    gap: 1D array, each element is the max-min of the corresponding column in data
    '''
    (m, n) = data.shape
    if min is None or gap is None:        
        min = np.min(data, 0)
        max = np.max(data, 0)        
        gap = max - min
        
    min_rep = np.tile(min, (m, 1))        
    gap_rep = np.tile(gap, (m, 1))
    return (data - min_rep) / gap_rep, min, gap

def un_normalize_zero_one(data, min, gap):
    '''
    un-normalize the normalized data. This is used for visualization purpose
       data : 2D array, each row is one data
    min   : 1D array, each element is the min of the corresponding column in data
    gap: 1D array, each element is the max-min of the corresponding column in data
    '''
    (m, n) = data.shape
    min_rep = np.tile(min, (m, 1))
    gap_rep = np.tile(gap, (m, 1))
    return np.multiply(data, gap_rep) + min_rep

def normalize_minus_one(data, min=None, gap=None):
    ''' [-1,1]
    normalize data to range [-1,1], also for normalizing targets if activation function is tanh()
    data : 2D array, each row is one data

    min   : 1D array, each element is the min of the corresponding column in data
    gap: 1D array, each element is the max-min of the corresponding column in data
    '''
    zero_one, min, gap = normalize_zero_one(data, min, gap)
    return 2.*zero_one - 1, min, gap
    
def un_normalize_minus_one(data, min, gap):
    '''
    un-normalize the normalized data. This is used for visualization purpose
       data : 2D array, each row is one data
    min   : 1D array, each element is the min of the corresponding column in data
    gap: 1D array, each element is the max-min of the corresponding column in data
    '''
    (m, n) = data.shape
    max = min + gap
    max_rep = np.tile(max, (m, 1))
    
    return 0.5 * (un_normalize_zero_one(data, min, gap) + max_rep)

def normalize_identity(data, min=None, gap=None):
    return data, min, gap


def un_normalize_identity(data, min=None, gap=None):
    return data
    
    
normalizer_name_map = {  # the mapping for data/label normalization
                       "None":(normalize_identity, un_normalize_identity),
                       "Gaussian":(normalize, un_normalize),  # usually for data normalizer
                       "Uniform_Minus_One":  (normalize_minus_one, un_normalize_minus_one),  # usually for both data and target normalizer only
                       "Uniform_Zero_One":  (normalize_zero_one, un_normalize_zero_one)  # for target normalizer only
                       }

target_activation_normalizer_map = {  # output activation with cooresponding target normalizer
                       "Sigmoid": (normalize_zero_one, un_normalize_zero_one),
                       "Softmax": (normalize_identity, un_normalize_identity),
                       "Tanh":    (normalize_minus_one, un_normalize_minus_one),
                       "LeTanh": (normalize_minus_one, un_normalize_minus_one),
                       "Linear": (normalize_minus_one, un_normalize_minus_one)
                       }

def flattenFeatures(features):
    
    a = array(features[0])
    n = len(features)
    for i in range(1, n):
        a = append(a, features[i], axis=1)
    return a

def num_mistakes(targetsMB, outputs):
    return (outputs.argmax(axis=1) != targetsMB.argmax(axis=1)).sum()

def sum_absolute_error(targets, outputs):
    if not isinstance(outputs, np.ndarray):
        outputs = outputs.as_numpy_array()
    if not isinstance(targets, np.ndarray):
        targets = targets.as_numpy_array()    
    return np.abs(outputs - targets).sum()

def mean_absolute_error(targets, outputs):
    return sum_absolute_error(targets, outputs) / targets.shape[0]

def sum_square_error(targets, outputs):  
    return (outputs - targets).euclid_norm() ** 2

def mean_square_error(targets, outputs):
    return sum_square_error(targets, outputs) / targets.shape[0]

metric_map = {
              "None":None,
              "sum_absolute_error": sum_absolute_error,
              "sum_square_error": sum_square_error,
              "num_mistakes": num_mistakes
              }
    # mae  = 0.0
    # for i in range(targets.shape[0]):
    # mae += mae*i/(i+1) + 1.0 * math.fabs(outputs[i]-targets[i])/(i+1)
         
    # return mae

def load_tsv(filename, delimiter = '\t'):
    return pd.read_csv(filename, sep = delimiter,error_bad_lines=False)

def save_tsv(filename, data, names = None, delimiter = '\t'):
    writer = csv.writer(open(filename, "w"), lineterminator="\n", delimiter=delimiter)
    
    if(names is not None):
        writer.writerow(names)
        
    writer.writerows(data)
    
