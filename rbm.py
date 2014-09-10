# this is a modified version of the example script that comes with cudamat
import time
from cudamat import gnumpy as gp
import sys
import numpy as np
import math
from common import *
import cPickle as pickle
from activation import *

def CD1(visiable, weights, visible_bias, hidden_bias, visible_unit = Binary(), hidden_unit = Binary(), dropout=None):
    """
    Using Gaussian hidden units hasn't been tested. By assuming the
    visible units are Binary, ReLU, or Gaussian and the hidden units
    are Binary or ReLU this function becomes quite simple.
    """
    #Positive phase    
    hidden = hidden_unit.activate(gp.dot(visiable, weights) + hidden_bias)    
    hidden_sampled = hidden_unit.sampleStates(hidden)

    #Negative phase
    v2 = visible_unit.activate(gp.dot(hidden_sampled, weights.T) + visible_bias)
    h2 = hidden_unit.activate(gp.dot(v2, weights) + hidden_bias)
    #calculate gradients
    gw = gp.dot(visiable.T, hidden) - gp.dot(v2.T, h2)
    gv = visiable.sum(axis=0) - v2.sum(axis=0)
    gh = hidden.sum(axis=0) - h2.sum(axis=0)

    return gw, gh, gv, v2
#dropout finetuning of non-dropout pre-trained DBN/DBMs is common practise 
def CD1_dropout(visiable, weights, visible_bias, hidden_bias, visible_unit = Binary(), hidden_unit = Binary(),dropout=0.0):
    """
    Using Gaussian hidden units hasn't been tested. By assuming the
    visible units are Binary, ReLU, or Gaussian and the hidden units
    are Binary or ReLU this function becomes quite simple.
    """
    #Positive phase    
    if(dropout == 0):
        hidden = hidden_unit.activate(gp.dot(visiable, weights) + hidden_bias)
    else:
        mask = gp.rand(*weights.shape) > dropout
        dropoutMultiplier = 1.0/(1.0-dropout)
        hidden = hidden_unit.activate(gp.dot(dropoutMultiplier*visiable, mask * weights) + hidden_bias)
                
    hidden_sampled = hidden_unit.sampleStates(hidden)

    #Negative phase
    v2 = visible_unit.activate(gp.dot(hidden_sampled, weights.T) + visible_bias)
    h2 = hidden_unit.activate(gp.dot(v2, weights) + hidden_bias)
    #calculate gradients
    gw = gp.dot(visiable.T, hidden) - gp.dot(v2.T, h2)
    gv = visiable.sum(axis=0) - v2.sum(axis=0)
    gh = hidden.sum(axis=0) - h2.sum(axis=0)

    return gw, gh, gv, v2

class rbm_cpu(object):
    #cpu model for persistent on disk, keep minimal data needed for prediction
    def __init__(self,gpu_model):
        self.weights = gpu_model.weights.as_numpy_array()
        self.hidden_bias = gpu_model.hidden_bias.as_numpy_array()
        self.visible_bias = gpu_model.visible_bias.as_numpy_array()
        
        self.visible_unittype_name = str(gpu_model.visible_unittype)
        self.hidden_unittype_name = str(gpu_model.hidden_unittype)
        
    def save(self,model_file):
        pickle.dump(self,open(model_file, 'wb'))
    
    @classmethod
    def load(cls,model_file):        
        return pickle.load(open(model_file,'r'))
    
    def predict(self, data):
        '''Compute feature with CPU'''
                        
        hid = np.dot(data, self.weights) + self.hidden_bias
        
        return activation_name_map[self.hidden_unittype_name+"_cpu"].activate(hid)
                
class rbm_gpu(object):
    '''A restricted boltzmann machine is a type of neural network auto-encoder.
    '''
    def __init__(self, num_hidden, num_visible, data=None, binary_visible=True, hidden_activation_function = Sigmoid(), dropout=0.0):
        '''Initialize a restricted boltzmann machine on GPU

        num_visible: The number of visible units. data.shape[0]
        num_hidden: The number of hidden units.
        binary: True if the visible units are binary, False if the visible units
          are normally distributed.
        '''
        self.num_hidden = num_hidden
        self.num_visible = num_visible 
        
        
        self.hidden_unittype = hidden_activation_function
        self.visible_unittype = self.hidden_unittype if binary_visible else Gaussian()
        
        if(data is not None):
            self.data = data#isinstance(data, gp.garray) and data or gp.garray(data.astype(np.float32))        
            self.num_visible = data.shape[1]
        
        self.weights = 0.1 * gp.randn(self.num_visible,self.num_hidden)
        self.hidden_bias = -4. * gp.ones(self.num_hidden)
        self.visible_bias = gp.zeros(self.num_visible)
    
        self.grad_weights = gp.zeros(self.weights.shape)
        self.grad_visible = gp.zeros(self.visible_bias.shape)
        self.grad_hidden = gp.zeros(self.hidden_bias.shape)
        self.dropout = dropout
        if(dropout is None or dropout==0):
            self.cd1 = CD1
        else:
            self.cd1=CD1_dropout
            
    @classmethod
    def get_gpu_model(cls,cpu_model):
        
        num_visible = cpu_model.weights.shape[0]
        num_hidden = cpu_model.weights.shape[1]
        rbm = rbm_gpu(num_hidden,num_visible)
        rbm.weights = gp.garray(cpu_model.weights.astype(np.float32))
        rbm.hidden_bias = gp.garray(cpu_model.hidden_bias.astype(np.float32))
        rbm.visible_bias = gp.garray(cpu_model.visible_bias.astype(np.float32))
        
        rbm.visible_unittype = activation_name_map[cpu_model.hidden_unittype_name]
        rbm.hidden_unittype = activation_name_map[cpu_model.hidden_unittype_name]
                
        return rbm
    
    def cpu_model(self):
        return rbm_cpu(self)

    def hidden_expectation(self, visible, bias=0.):
        '''Given visible data, return the expected hidden unit values.'''
        hidden = gp.dot(visible, self.weights) + self.hidden_bias + bias
        return self.hidden_unittype.activate(hidden)

    def visible_expectation(self, hidden, bias=0.):
        '''Given hidden states, return the expected visible unit values.'''
        visible = gp.dot(hidden, self.weights.T) + self.visible_bias + bias        
        return self.visible_unittype.activate(visible)

    def predict(self, minibatchStream, asNumpy = False):
        '''
        Given visible data, return the expected hidden unit values,run in batch mode to save GPU memory
        data: data from the layer below this layer     
        '''
        for minibatch,_ in minibatchStream:
            outputActs = self.hidden_expectation(minibatch)
            yield outputActs.as_numpy_array() if asNumpy else outputActs
            
        
    def train_one_batch(self, batch, l2_reg, learnRate):
        '''
        '''
        batch_size = batch.shape[0]
        
        visible =  isinstance(batch, gp.garray) and batch or gp.garray(batch.astype(np.float32)) 
        
        gw, gh, gv, v2 = self.cd1(visible, self.weights, self.visible_bias, self.hidden_bias, self.visible_unittype, self.hidden_unittype,self.dropout)
                
        self.grad_weights = self.momentum*self.grad_weights + gw
        self.grad_visible = self.momentum*self.grad_visible + gv
        self.grad_hidden = self.momentum*self.grad_hidden + gh

        if l2_reg > 0:
            self.weights *= 1 - l2_reg*learnRate
        
        self.weights += (learnRate/batch_size) * self.grad_weights
        self.visible_bias += (learnRate/batch_size) * self.grad_visible
        self.hidden_bias += (learnRate/batch_size) * self.grad_hidden

        #we compute squared error even for binary visible unit RBMs
        return (v2-visible).euclid_norm()**2/(batch.shape[0]*batch.shape[1])
    
    def train(self, minibatch_stream, epochs, mbPerEpoch,  momentum=0.9, l2_reg=0, learn_rate=0.2):
        """Train one rbm with the self.data and parameters"""
        self.momentum = momentum
        
    
        start_time = time.time()
       
        for e in range(epochs):
            errors = []
            
            epoch_start_time = time.time()
            for j in range(mbPerEpoch):
                minibatch, _ = minibatch_stream.next()
                #this is what i got from test_rbm.py
                #learnRate = alpha * math.exp(-i/tau) #learning rate keep decreasing slowly
                #learning, the weights and biases already updated in this function                
                err = self.train_one_batch(minibatch, l2_reg, learn_rate)
                
                errors.append(err)
            print 'Epoch '+str(e+1)+'/'+str(epochs)+'\tError '+str(np.mean(errors)) +'\tSeconds '+ str(time.time() - epoch_start_time)           
       
        print "RBM training seconds: " + str(time.time() - start_time)
    
def train_rbm(layers_config,layer=1):
    
    if(layer==0): pass #input layer, nothing to do
    
    config = layers_config[layer]
    inputconfig = layers_config[0]

    if(config['type'] != 'rbm'): pass
    
    batch_size = inputconfig['batch_size']
    
    if(layer==1): training_data = np.load(inputconfig['train_data'])
    else: training_data = np.load(layers_config[layer-1]['output_data'])
    
    num_visible = training_data.shape[1]
    
    binary_visible = True if layer != 1 else inputconfig['binary_input']
    if 'binary_input' in config: binary_visible = config['binary_input']  

    kwargs = dict(num_hidden=config['hidden_nodes'], num_visible=num_visible, data=training_data, binary_visible = binary_visible)
    
    rbm = rbm_gpu(**kwargs)
    
    kwargs = dict(batch_size= batch_size, epoch=config['epoch'], alpha=config['alpha'], \
                  momentum=config['momentum'],target_sparsity=config['sparsity'],l2_reg=config['l2_regularization'],tau=config['tau'])
    
    rbm.train(**kwargs)
    
    rbm.cpu_model().save(config['model_file'])
    
    predicted = rbm.predict(batch_size= batch_size)
    
    np.save(config['output_data'],predicted)
    ##DONE##

####Testing###
if __name__ == '__main__':
    
    training_data = np.array([[1,1,1,0,0,0],[1,0,1,0,0,0],[1,1,1,0,0,0],[0,0,1,1,1,0], [0,0,1,1,0,0],[0,0,1,1,1,0],[0,0,1,1,0,0],[0,0,1,1,1,0]]) #8 x 6
    
    rbm = rbm_gpu(num_hidden=2,num_visible=None,data = training_data, binary_visible = True)
    
    minibatch_stream = gpu_buffer(data=training_data,label=None, batch_size = 2, randomize = True, gpu_buffer_size_MB=200).iter_minibatch()

    rbm.train(minibatch_stream, epochs=3, mbPerEpoch=4,  momentum=0.9, l2_reg=0, learn_rate=0.1)

    print rbm.weights
    
    minibatch_stream = gpu_buffer(data=training_data,label=None, batch_size = 2, randomize = False, gpu_buffer_size_MB=200).iter_minibatch()
    for predicted in rbm.predict(minibatch_stream):
        print "Predicted:" + str(predicted)
    
    
    user = np.array([[0,0,0,1,1,0]])
    for predicted in rbm.predict([(user,None)]):
        print "Predicted:" + str(predicted)
