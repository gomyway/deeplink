import common
import math
from cudamat import gnumpy as gp
'''
    Define different loss functions to be used in supervised neural network training
    It is different from hidden node's activation function which is simply adding nonlinearity and normalization for training
'''
class CrossEntropyLoss(object):
    def __init__(self, epsilon=1e-15):
        self.epsilon = float(epsilon)
        
    def loss(self, t, y):
        '''loss = -sum(t*logy + (1-t)*log(1-y))'''
        y[y<self.epsilon].assign(self.epsilon)
        y[y>1-self.epsilon].assign(1 - self.epsilon)
                
        return -(t * y.log() + (1-t) * (1-y).log()).sum()
    
    def d_loss_wrt_y(self, t, y):
        y[y<self.epsilon].assign(self.epsilon)
        y[y>1-self.epsilon].assign(1 - self.epsilon)
                
        return -t/y + (1-t)/(1-y)
    
    def __str__(self):
        return "CrossEntropyLoss"
    

class MultiClassLogLoss(object):
    '''log loss = -\frac{1}{N}\sum_{i=1}^N\sum_{j=1}^M y_{i,j}\log(p_{i,j})
        softmax output should use this instead of CrossEntropyLoss above
    '''
    def __init__(self, epsilon=1e-5):
        self.epsilon = float(epsilon)
        
    def loss(self, t, y):
        '''loss = -avg(t*logy)
        Multi class version of Logarithmic Loss metric.
        https://www.kaggle.com/wiki/MultiClassLogLoss
    
        idea from this post:
        http://www.kaggle.com/c/emc-data-science/forums/t/2149/is-anyone-noticing-difference-betwen-validation-and-leaderboard-error/12209#post12209
        https://www.kaggle.com/c/predict-closed-questions-on-stack-overflow/forums/t/2644/multi-class-log-loss-function
        Parameters
        ----------
        t_true : array, shape = [n_samples, n_classes]
        y_pred : array, shape = [n_samples, n_classes]
    
        Returns
        -------
        loss : float
        '''
        y[y<self.epsilon].assign(self.epsilon)
        y[y>1-self.epsilon].assign(1 - self.epsilon)
    
        # normalize row sums to 1
        y /= y.sum(axis=1)
    
        rows = t.shape[0]
        vsota = (t * y.log()).sum()
        return -1.0 * vsota / rows
    
    def d_loss_wrt_y(self, t, y):
        y[y<self.epsilon].assign(self.epsilon)
        y[y>1-self.epsilon].assign(1 - self.epsilon)
                
        return -t/y + (1-t)/(1-y)
    
    def __str__(self):
        return "CrossEntropyLoss"
    
class SquareLoss(object):
    def loss(self, t, y):
        '''loss = 1/2 * sum (t-y)^2'''
        return 0.5*((y-t).euclid_norm()**2)
        
    def d_loss_wrt_y(self,t, y): #error
        return y - t
     
    def __str__(self):
        return "SquareLoss"
    
class AbsoluteLoss(object):
    '''Use y=sqrt(x^2+epsilon) as an approximation, epsilon controls how smooth around x=0'''
    def __init__(self,epsilon = 0.01):
        self.epsilon = epsilon
        
    def loss(self, t, y):
        '''Loss = sum|t-y|'''
        #return (t-y).abs().sum()
        return math.sqrt((y-t).euclid_norm()**2 + self.epsilon)
        
    def d_loss_wrt_y(self, t, y):
        '''Use y=sqrt(x^2+epsilon) as an approximation'''
        return (y-t)/self.loss(t, y)
        #return (y>t) - (y<t)         
    
    def __str__(self):
        return "AbsoluteLoss"


loss_name_map = {
                 "CrossEntropyLoss":CrossEntropyLoss(), 
                 "MultiClassLogLoss":MultiClassLogLoss(),
                 "SquareLoss":    SquareLoss(),
                 "AbsoluteLoss":AbsoluteLoss()
                 }
    