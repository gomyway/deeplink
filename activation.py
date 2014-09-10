import common
import numpy as np
from cudamat import gnumpy as gp

class Sigmoid(object):
    def activate(self, netInput):
        return netInput.sigmoid()
    def dEdNetInput(self, acts):
        return acts*(1-acts)
    def error(self, targets, netInput, acts = None):
        #return (targets*logOnePlusExp(-netInput) + (1-targets)*logOnePlusExp(netInput)).sum()
        #return (logOnePlusExp(netInput)-targets*netInput).sum()
        return (netInput.log_1_plus_exp()-targets*netInput).sum()
    def HProd(self, vect, acts):
        return vect*acts*(1-acts)
    def dErrordNetInput(self, targets, netInput, acts = None):
        if acts == None:
            acts = self.activation(netInput)
        return acts - targets
    #only for RBM
    def sampleStates(self, acts):
        return gp.rand(*acts.shape) <= acts
        
    def __str__(self):
        return "Sigmoid"

class Sigmoid_cpu(object):
    def activate(self, netInput):
        return 1./(1. + np.exp(-netInput))
    def __str__(self):
        return "Sigmoid_cpu"

#You can write tanh in terms of sigmoid.
#def tanh(ar):
#    return 2*(2*ar).sigmoid()-1
# There might be a "better" tanh to use based on Yann LeCun's
# efficient backprop paper is in 1.7159 * tanh ( 2 * x/3).
class Tanh(object):
    def activate(self, netInput):
        return gp.tanh(netInput)
    def dEdNetInput(self, acts):#1 - tanh^2(x)  
        return 1-acts*acts
    def error(self, targets, netInput, acts = None):
        #return (targets*logOnePlusExp(-netInput) + (1-targets)*logOnePlusExp(netInput)).sum()
        #return (logOnePlusExp(netInput)-targets*netInput).sum()
        return ((2.* netInput).log_1_plus_exp() - netInput - targets*netInput).sum()
        
    def dErrordNetInput(self, targets, netInput, acts = None):
        if acts == None:
            acts = self.activation(netInput)
        return acts - targets
    def __str__(self):
        return "Tanh"

class Tanh_cpu(object):
    def activate(self, netInput):
        return np.tanh(netInput) #2./(1. + np.exp(-2. * netInput)) - 1.
    def __str__(self):
        return "Tanh_cpu"

class LeTanh(object):
    '''based on Yann LeCun's
        efficient backprop paper, best activation funcion is f(x) = 1.7159 * tanh ( 2 * x/3)'''    
    def __init__(self, a = 1.7159, b=0.66667): #1.7159 * tanh ( 2 * x/3)
        self.a = a
        self.b = b

    def activate(self, netInput):
        return self.a * gp.tanh(self.b * netInput)
    
    def dEdNetInput(self, acts):#1 - tanh^2(x)  
        return self.a*self.b - (self.b/self.a)* acts*acts
    
    def error(self, targets, netInput, acts = None):
        #return (targets*logOnePlusExp(-netInput) + (1-targets)*logOnePlusExp(netInput)).sum()
        #return (logOnePlusExp(netInput)-targets*netInput).sum()
        return ((self.a/self.b)*(2.*self.b* netInput).log_1_plus_exp() - self.a*netInput - targets*netInput).sum()
        
    def dErrordNetInput(self, targets, netInput, acts = None):
        if acts == None:
            acts = self.activation(netInput)
        return acts - targets
    def __str__(self):
        return "LeTanh"

class LeTanh_cpu(object):
    def __init__(self, a = 1.7159, b=0.66667): #1.7159 * tanh ( 2 * x/3)
        self.a = a
        self.b = b            
    def activate(self, netInput):
        return sef.a*np.tanh(self.b*netInput)
    def __str__(self):
        return "Tanh_cpu"
    
        
#rectified linear unit (ReLU) f(x) = max(0,x) 
class ReLU(object):
    def __init__(self, krizNoise = False, cap=None):
        self.krizNoise = krizNoise
        self.cap = None    
    def activate(self, netInput):
        y = netInput*(netInput > 0) #max(0,x)
        if(self.cap is not None): # min(self.cap,y)
            y = y * (y<=self.cap) + self.cap * (y>self.cap)
        return y
        #return netInput.log_1_plus_exp()
    
    def dEdNetInput(self, acts):
        delta = (acts > 0)
        if(self.cap is not None): #min(cap,delta)
            delta = delta * (delta<=self.cap) 
        return delta
    
        #return 1. - 1./acts.exp()
    def sampleStates(self, acts):
        if self.krizNoise:
            return self.activate(acts + gp.randn(*acts.shape))
        tiny = 1e-30
        stddev = gp.sqrt(acts.sigmoid() + tiny)
        return self.activate( acts + stddev*gp.randn(*acts.shape) )    
    def __str__(self):
        return "ReLU"
#rectified linear unit (ReLU) f(x) = max(0,x), uses its smooth approximate f(x) = log(1+ exp(x))
class ReLU_smooth(object):
    def __init__(self, krizNoise = False,cap=None):
        self.krizNoise = krizNoise    
        self.cap = cap
    def activate(self, netInput):
        #return netInput*(netInput > 0) #max(0,x)
        y = netInput.log_1_plus_exp()
        if(self.cap is not None): #min(cap,y)
            y = y * (y<=self.cap) + self.cap * (y>self.cap)
        return y
    
    def dEdNetInput(self, acts):
        #return acts > 0
        delta = (1. - 1./acts.exp())
        
        if(self.cap is not None):
            delta = delta * (delta<=self.cap)
        return delta
    
    def sampleStates(self, acts):
        if self.krizNoise:
            return self.activate(acts + gp.randn(*acts.shape))
        tiny = 1e-30
        stddev = gp.sqrt(acts.sigmoid() + tiny)
        return self.activate( acts + stddev*gp.randn(*acts.shape) )    
    def __str__(self):
        return "ReLU"
    
class Linear(object):
    def activate(self, netInput):
        return netInput
    def dEdNetInput(self, acts):
        return 1 #perhaps returning ones(acts.shape) is more appropriate?
    def error(self, targets, netInput, acts = None):
        diff = targets-netInput
        return 0.5*(diff*diff).sum()
    def HProd(self, vect, acts):
        return vect
    def dErrordNetInput(self, targets, netInput, acts = None):
        if acts == None:
            acts = self.activation(netInput)
        return acts - targets
    def __str__(self):
        return "Linear"

class Softmax(object):
    def activate(self, netInput):
        Zshape = (netInput.shape[0],1)
        acts = netInput - netInput.max(axis=1).reshape(*Zshape)
        acts = acts.exp()
        return acts/acts.sum(axis=1).reshape(*Zshape)
    def HProd(self, vect, acts):
        return acts*(vect-(acts*vect).sum(1).reshape(-1,1))
    def dErrordNetInput(self, targets, netInput, acts = None):
        if acts == None:
            acts = self.activation(netInput)
        return acts - targets
    def error(self, targets, netInput, acts = None):
        ntInpt = netInput - netInput.max(axis=1).reshape(netInput.shape[0],1)
        logZs = ntInpt.exp().sum(axis=1).log().reshape(-1,1)
        err = targets*(ntInpt - logZs)
        return -err.sum()
    def __str__(self):
        return "Softmax"  

class Softmax_cpu(object):
    def activate(self, netInput):
        Zshape = (netInput.shape[0],1)
        acts = netInput - netInput.max(axis=1).reshape(*Zshape)
        acts = np.exp(acts)
        return acts/acts.sum(axis=1).reshape(*Zshape)
                
    def __str__(self):
        return "Softmax_cpu"

#For RBM binary input/hidden units
class Binary(object):
    def activate(self, netInput):
        return netInput.sigmoid()
    def sampleStates(self, acts):
        return gp.rand(*acts.shape) <= acts
    def __str__(self):
        return "Binary"  

#For RBM real number input units,hidden nodes can be either Binary or ReLU
class Gaussian(object):
    def activate(self, netInput):
        return netInput
    def sampleStates(self, acts): #probably shouldn't use this
        #return acts + gp.randn(*acts.shape)
        raise Exception("Gaussian.sampleStates() Should not be used")
    def __str__(self):
        return "Gaussian"  

activation_name_map = {#the mapping for cpu version activate function purpose only
                       "Sigmoid_cpu":Sigmoid_cpu(), #activation is different in numpy than gnumpy
                       "Sigmoid":    Sigmoid(),
                       "Tanh_cpu":Tanh_cpu(),
                       "Tanh":Tanh(),
                       "LeTanh":LeTanh(),
                       "LeTanh_cpu":LeTanh_cpu(),
                       "ReLU_cpu":ReLU(),
                       "ReLU":ReLU(),
                       "ReLU_cap":ReLU(cap=6.0),
                       "ReLU_smooth":ReLU_smooth(),
                       "ReLU_smooth_cap":ReLU_smooth(cap=6.0),
                       "Linear_cpu":Linear(),
                       "Linear":Linear(),
                       "Binary_cpu":Sigmoid_cpu(),
                       "Binary":Binary(),                       
                       "Gaussian_cpu":Gaussian(),
                       "Gaussian":Gaussian(),
                       "Softmax_cpu":Softmax_cpu(),
                       "Softmax":Softmax()
                       }
