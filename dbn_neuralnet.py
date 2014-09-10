
import numpy as np
import cPickle as pickle
import time
import os
import h5py

from common import *
from activation import *
from cudamat import gnumpy as gp
from rbm import *
from lossfunction import *

class dbn_neuralnet_cpu(object):
    '''
    Objects of this class hold values of the RBMs in numpy arrays to free up space 
    on the GPU
    '''
    def __init__(self, dbn):        
        self.weights = numpyify(dbn.weights)
        self.biases = numpyify(dbn.biases)
        self.hidden_activation_names = [str(activition) for activition in dbn.hidden_activation_functions]
        self.output_activation_functionName = str(dbn.output_activation_function)
    
    def save(self,model_file):
        pickle.dump(self,open(model_file, 'wb'))
    
    @classmethod
    def load(cls,model_file):        
        return pickle.load(open(model_file,'r'))
            
    def forward_propagate(self, data ):
        """
        Perform a forward pass through the network. Updates state which holds the input followed by each hidden layer's activation and
        finally the net input incident on the output layer. We return the actual output unit activations
        """
        #data = data.as_numpy_array() if isinstance(data, gp.garray) else data
        
        num_layers = len(self.weights)
        #self.state holds everything before the output nonlinearity, including the net input to the output units
        state = [data]
        for i in range(num_layers - 1):
            curActs = self.hidden_activation_functions[i].activate(np.dot(state[-1], self.weights[i]) + self.biases[i])
            state.append(curActs)
        
        state.append(np.dot(state[-1], self.weights[-1]) + self.biases[-1])
        acts = self.output_activation_function.activate(state[-1])
        
        return acts

    def predict(self, data):     
        num_layers = len(self.weights)  
                
        self.hidden_activation_functions = [activation_name_map[str_activation+"_cpu"] for str_activation in self.hidden_activation_names]                
        self.output_activation_function = activation_name_map[self.output_activation_functionName+"_cpu"] 
        
        output_activations = self.forward_propagate(data)
        return output_activations

class dbn_neuralnet_gpu(object):
    '''
        Train multi-layer NeuralNet using initial weights from pre-trained RBMs
        binary_visible can be False only for first layer with real value inputs, other layers always see binary inputs
    '''
    def __init__(self, initialWeights, initialBiases,hidden_activation_function=Sigmoid(),binary_visible = False):
        
        num_layers = len(initialWeights)
        
        self.binary_visible = binary_visible
        self.learn_rate_decay_half_life = 100
        self.apply_L2cost_after = 100
        self.weights = garrayify(initialWeights)
        self.biases = garrayify(initialBiases)
        
        self.hidden_activation_functions = [hidden_activation_function for i in range(num_layers)]
        
        #self.use_ReLU = use_ReLU
        #if use_ReLU:
        #    self.hidden_activation_functions = [ReLU() for i in range(num_layers - 1)]
        #else:
        #    self.hidden_activation_functions = [Sigmoid() for i in range(num_layers - 1)]
        
        #state variables modified in backward_propagate
        self.grad_weights = [gp.zeros(self.weights[i].shape) for i in range(num_layers)]
        self.grad_bias = [gp.zeros(self.biases[i].shape) for i in range(num_layers)]

    @classmethod
    def get_gpu_model(cls,cpu_model):                

        dbn = dbn_neuralnet()
        
        dbn.weights = garrayify(cpu_model.weights)
        dbn.biases = garrayify(cpu_model.biases)
        
        dbn.output_activation_function = activation_name_map[cpu_model.output_activation_functionName]
        dbn.hidden_activation_functions = [activation_name_map[name] for name in cpu_model.hidden_activation_names]
            
        return dbn
    
    def cpu_model(self):
        return dbn_neuralnet_cpu(self)
    
    def forward_propagate(self, inputBatch, upToLayer = None ):
        """
        Perform a (possibly partial) forward pass through the
        network. Updates self.state which, on a full forward pass,
        holds the input followed by each hidden layer's activation and
        finally the net input incident on the output layer. For a full
        forward pass, we return the actual output unit activations. In
        a partial forward pass we return None.
        """
        if upToLayer == None: #work through all layers
            upToLayer = len(self.weights)
        #self.state holds everything before the output nonlinearity, including the net input to the output units
        self.state = [inputBatch]
        for i in range(min(len(self.weights) - 1, upToLayer)):
            curActs = self.hidden_activation_functions[i].activate(gp.dot(self.state[-1], self.weights[i]) + self.biases[i])
            self.state.append(curActs)
        if upToLayer >= len(self.weights):
            self.state.append(gp.dot(self.state[-1], self.weights[-1]) + self.biases[-1])
            self.acts = self.output_activation_function.activate(self.state[-1])
            return self.acts
        #we didn't reach the output units
        # To return the first set of hidden activations, we would set
        # upToLayer to 1.
        return self.state[upToLayer]
    
    def forward_propagate_dropout(self, inputBatch, upToLayer = None ):
        """
        Perform a (possibly partial) forward pass through the
        network. Updates self.state which, on a full forward pass,
        holds the input followed by each hidden layer's activation and
        finally the net input incident on the output layer. For a full
        forward pass, we return the actual output unit activations. In
        a partial forward pass we return None.
        
        reference: 
        IMPROVING DEEP NEURAL NETWORKS FOR LVCSR USING RECTIFIED LINEAR UNITS AND DROPOUT
        """
        if upToLayer == None: #work through all layers
            upToLayer = len(self.weights)
        
        self.state = [inputBatch]
        for i in range(min(len(self.weights) - 1, upToLayer)):            
            if self.dropouts[i] > 0: 
                mask = gp.rand(*self.weights[i].shape) > self.dropouts[i]
                dropoutMultiplier = 1.0/(1.0-self.dropouts[i])
                curActs = self.hidden_activation_functions[i].activate(gp.dot(dropoutMultiplier*self.state[-1], mask * self.weights[i]) + self.biases[i])
            else:
                curActs = self.hidden_activation_functions[i].activate(gp.dot(self.state[-1], self.weights[i]) + self.biases[i])
            #apply dropout on hidden units
            #if self.dropouts[i+1] > 0: curActs = curActs * (gp.rand(*curActs.shape) > self.dropouts[i+1])             
            self.state.append(curActs)
        
        if upToLayer >= len(self.weights):
            self.state.append(gp.dot(self.state[-1], self.weights[-1]) + self.biases[-1])
            self.acts = self.output_activation_function.activate(self.state[-1])            
            return self.acts
        #we didn't reach the output units
        # To return the first set of hidden activations, we would set
        # upToLayer to 1.
        return self.state[upToLayer]
    

    def backward_propagate(self, outputErrSignal, forward_propagateState = None):
        """
        Perform a backward pass through the network. forward_propagateState
        defaults to self.state (set during forward_propagate) and outputErrSignal
        should be self.output_activation_function.dErrordNetInput(...).
        """
        
        if forward_propagateState == None:
            forward_propagateState = self.state
        assert(len(forward_propagateState) == len(self.weights) + 1)

        errSignals = [None for i in range(len(self.weights))]
        errSignals[-1] = outputErrSignal
        for i in reversed(range(len(self.weights) - 1)):
            errSignals[i] = gp.dot(errSignals[i+1], self.weights[i+1].T)*self.hidden_activation_functions[i].dEdNetInput(forward_propagateState[i+1])
        return errSignals

    def gradients(self, forward_propagateState, errSignals):
        """
        Lazily generate (negative) gradients for the weights and biases given
        the result of forward_propagate (forward_propagateState) and the result of backward_propagate
        (errSignals).
        """
        assert(len(forward_propagateState) == len(self.weights)+1)
        assert(len(errSignals) == len(self.weights) == len(self.biases))
        for i in range(len(self.weights)):
            yield gp.dot(forward_propagateState[i].T, errSignals[i]), errSignals[i].sum(axis=0)   

            
    def forward_backward_propagate(self, inputBatch, targetBatch):
        ''' back propagate to get the error signals for each hidden layer
            the error signal for output unit j:  delta_j = - d E/d net_j  = t-y for sum-squared loss E = 1/2 sum (t - y)^2 
            http://www.willamette.edu/~gorr/classes/cs449/backprop.html
        '''
        output_activations = self.forward_propagate_dropout(inputBatch)

        outputErrSignal = -self.loss_function.d_loss_wrt_y(targetBatch, output_activations)
        error = self.loss_function.loss(targetBatch, output_activations)
        
        errSignals = self.backward_propagate(outputErrSignal)
        return errSignals, output_activations, error
    
  
    def step(self, inputBatch, targetBatch, learn_rates, momentum, L2_costs):
        minibatch_size = inputBatch.shape[0]
        
        errSignals, output_activations, error = self.forward_backward_propagate(inputBatch, targetBatch)

        for i in range(len(self.weights)):
            self.grad_weights[i] *= momentum
            self.grad_bias[i] *= momentum
            
        for i, (weightGrad, biasGrad) in enumerate(self.gradients(self.state, errSignals)):
            self.grad_weights[i] += (learn_rates[i]/minibatch_size)*(weightGrad - L2_costs[i]*self.weights[i])
            self.grad_bias[i] += (learn_rates[i]/minibatch_size)*biasGrad
        
        for i in range(len(self.weights)):
            self.weights[i] +=  self.grad_weights[i]
            self.biases[i] +=  self.grad_bias[i]
            
        return error, output_activations

    
    def apply_updates(self, destWeights, destBiases, curWeights, curBiases, WGrads, biasGrads):
        for i in range(len(destWeights)):
            destWeights[i] = curWeights[i] + WGrads[i]
            destBiases[i] = curBiases[i] + biasGrads[i]
            
    def step_nesterov(self, inputBatch, targetBatch, learn_rates, momentum, L2_costs):
        minibatch_size = inputBatch.shape[0]
        
        curWeights = [w.copy() for w in self.weights]
        curBiases = [b.copy() for b in self.biases]
        
        for i in range(len(self.weights)):
            self.grad_weights[i] *= momentum
            self.grad_bias[i] *= momentum
                        
        self.apply_updates(self.weights, self.biases, curWeights, curBiases, self.grad_weights, self.grad_bias)
        
        errSignals, output_activations, error = self.forward_backward_propagate(inputBatch, targetBatch)
        
        #self.scaleDerivs(momentum)
        for i, (weightGrad, biasGrad) in enumerate(self.gradients(self.state, errSignals)):
            self.grad_weights[i] += learn_rates[i]*(weightGrad/minibatch_size - L2_costs[i]*self.weights[i])
            self.grad_bias[i] += (learn_rates[i]/minibatch_size)*biasGrad

        self.apply_updates(self.weights, self.biases, curWeights, curBiases, self.grad_weights, self.grad_bias)

        return error, output_activations
    
    def update_parameters(self, step):
        """
        learn_rate decay, momentum increase, L2_cost enable after some steps
        """
        momentum_change_steps = 20
        initial_momentum = 0.5
        momentum = self.momentum - (self.momentum - initial_momentum)*np.exp(-float(step)/momentum_change_steps)
        
        learn_rates = self.learn_rates
        
        learn_rates = [rate / (1 + float(step) / self.learn_rate_decay_half_life) for rate in learn_rates]
        
        
        L2_costs = self.L2_costs
        
        if(step<=self.apply_L2cost_after):
            L2_costs = [0. for i in self.L2_costs]
        
        return momentum, learn_rates, L2_costs

    def fine_tune(self, minibatch_stream, epochs, num_batches_per_epoch, momentum = 0.9, loss = None, learn_rates=[], L2_costs=[], dropouts=[], output_activation_function=Sigmoid(),\
                   loss_function = SquareLoss(), nesterov=False,learn_rate_decay_half_life=100,apply_L2cost_after=10):
        
        num_layers = len(self.weights)
        
        self.learn_rates = learn_rates
        self.learn_rate_decay_half_life = learn_rate_decay_half_life
        self.momentum = momentum
        self.L2_costs = L2_costs
        self.apply_L2cost_after = apply_L2cost_after,
        self.dropouts = dropouts
        self.output_activation_function = output_activation_function        
        step = self.step_nesterov if nesterov else self.step
        self.loss_function = loss_function
        
        print "Fine tune dbn with parameters: num_layers %d epoches %d learn_rates %s momentum %g L2_costs %s dropouts %s output_activation_function %s loss_function %s nesterov %s" \
                % (num_layers, epochs, str(learn_rates), momentum,str(L2_costs), str(dropouts), str(output_activation_function), str(loss_function), str(nesterov)) 
                                      
        for ep in range(epochs):
            start_time = time.time() 
            totalCases = 0
            sumErr = 0
            sumLoss = 0
            
            momentum, learn_rates, L2_costs = self.update_parameters(ep)
            
            for i in range(num_batches_per_epoch):
                inpMB,targMB = minibatch_stream.next()
                err,outMB = step(inpMB, targMB, learn_rates, momentum, L2_costs)
                sumErr += err
                if loss != None:
                    sumLoss += loss(targMB, outMB)
                totalCases += inpMB.shape[0]

            yield sumErr/float(totalCases), sumLoss/float(totalCases), time.time() - start_time           
     

    def predict(self, minibatchStream, asNumpy = False,uptoLayer=None):
        for inputBatch,_ in minibatchStream:
            #inputBatch = inpMB if isinstance(inpMB, gp.garray) else gp.garray(inpMB)
            output_activations = self.forward_propagate(inputBatch,uptoLayer)
            #error = (targetMB-output_activations).abs().sum() if targetMB is not None else None
            yield output_activations.as_numpy_array() if asNumpy else output_activations#,targetMB.as_numpy_array() if asNumpy else targetMB

     
    def preTrainIthRBM(self, i, minibatchStream, epochs, mbPerEpoch, learn_rate, momentum, L2_cost,dropout=None):

        num_visible,num_hidden = self.weights[i].shape
        
        binary_input_visible = (i==0 and self.binary_visible)
        
        #RBM training has to use ReLU_smooth to avoid overfitting too much
        hidden_activation_function= ReLU_smooth() if isinstance(self.hidden_activation_functions[i],ReLU) else self.hidden_activation_functions[i]
        
        rbm = rbm_gpu(num_hidden=num_hidden, num_visible=num_visible, data=None, binary_visible=binary_input_visible, hidden_activation_function=hidden_activation_function,dropout=dropout)
            
        rbm.train(minibatchStream, epochs, mbPerEpoch,  momentum=momentum, l2_reg=L2_cost, learn_rate=learn_rate)
        
        self.weights[i]=rbm.weights
        self.biases[i]=rbm.hidden_bias
        #self.dbn.visible_bias[i]=rbm.visible_bias #we just disgard visible_bias as it won't be used in neuralnet
        return rbm
    
    def unsupervised_pretrain(self,num_hidden_layers, data, batch_sizes=[],epochs=[],learn_rates=[],momentums=[],L2_costs=[], dropouts=[]):
        #randomly shuffle the data
        #idx = np.random.permutation(len(data))
        #data = data[idx]
        tempfile = "rbm.%.7f.h5" % time.time()
        hdf5_file = h5py.File(tempfile,'a')
        
        pretrain_start_time=time.time()        
        for i in range(num_hidden_layers): 
            print "Pretraining layer %d with parameters: hidden_nodes %d epoches %d learn_rate %g momentum %g L2_cost %g dropout %g" % (i+1, self.weights[i].shape[1], epochs[i], learn_rates[i],momentums[i],L2_costs[i],dropouts[i])       
            minibatch_stream = cpu_gpu_buffer(data=data,label=None, batch_size = batch_sizes[i], randomize = True, gpu_buffer_size_MB=256,cpu_buffer_size_MB=4096).iter_minibatch()
            
            nBatchesPerEpoch=int(np.ceil(1.0*data.shape[0]/batch_sizes[i]))
            
            layer_start_time = time.time()
            #train
            rbm=self.preTrainIthRBM(i,minibatch_stream,epochs[i],nBatchesPerEpoch,learn_rates[i],momentums[i],L2_costs[i],dropouts[i])
            #predict
            minibatch_stream = cpu_gpu_buffer(data=data,label=None, batch_size = batch_sizes[i], randomize = False, gpu_buffer_size_MB=256, cpu_buffer_size_MB=4096).iter_minibatch()            
            features = tuple(rbm.predict(minibatch_stream,asNumpy = True))  
            data = np.vstack(features)
            print "Generated features with dimension: %d,%d" % data.shape
            print "Pretrained layer %d with seconds: %g\n" % (i, time.time()-layer_start_time)
            name = "data%.7f" % time.time()
            hdf5_file.create_dataset(name, dtype='f4', data=data, chunks=(2, data.shape[1])) 
            data = hdf5_file[name]   
        print "Pretrain all layers with total seconds: %g" % (time.time()-pretrain_start_time)
        hdf5_file.close()
        os.remove(tempfile)
         
    def weightsDict(self):
        d = {}
        if len(self.weights) == 1:
            d['weights'] = np.empty((1,), dtype=np.object)
            d['weights'][0] = numpyify(self.weights)[0]
            d['biases'] = np.empty((1,), dtype=np.object)
            d['biases'][0] = numpyify(self.biases)[0]
        else:
            d['weights'] = np.array(numpyify(self.weights)).flatten()
            d['biases'] = np.array(numpyify(self.biases)).flatten()
            
        d['num_layers'] = len(self.weights)
        
        return d
    
    def loadWeights(self, path):
        fd = open(path, 'rb')
        d = np.load(fd)
        num_layers = d['num_layers']
        if num_layers > 1:
            self.weights[:num_layers] = garrayify(d['weights'].flatten())[:num_layers]
            self.biases[:num_layers] = garrayify(d['biases'].flatten())[:num_layers]
        else:
            self.weights = garrayify(d['weights'].flatten())
            self.biases = garrayify(d['biases'].flatten())

        fd.close()
    
    def saveWeights(self, path):
        np.savez(path, **self.weightsDict())
        
def initWeightMatrix(shape, scale, maxNonZeroPerColumn = None, uniform = False):
    #number of nonzero incoming connections to a hidden unit
    fanIn = shape[0] if maxNonZeroPerColumn==None else min(maxNonZeroPerColumn, shape[0])
    if uniform:
        W = scale*(2*np.random.rand(*shape)-1)
    else:
        W = scale*np.random.randn(*shape)
    for j in range(shape[1]):
        perm = np.random.permutation(shape[0])
        W[perm[fanIn:],j] *= 0
    return W


def buildNet(layerSizes, scales, fanOuts, binary_visible, hidden_activation_function = Sigmoid(), uniforms = None):
    shapes = [(layerSizes[i-1],layerSizes[i]) for i in range(1, len(layerSizes))]
    assert(len(scales) == len(shapes) == len(fanOuts))
    if uniforms == None:
        uniforms = [False for s in shapes]
    assert(len(scales) == len(uniforms))
    

    # randomize the network weights according to the Bottou proposition
    # this is borrowed from the ffnet project:
    # http://ffnet.sourceforge.net/_modules/ffnet.html#ffnet.randomweights
    n = 0
    for i in range(len(layerSizes)-1):
        n += layerSizes[i]*layerSizes[i+1]
        n += layerSizes[i+1]

    bound = 2.38 / np.sqrt(n)

    initialWeights = []
    for layer in range(len(shapes)):
       W = [np.random.uniform(-bound, bound) for i in range(shapes[layer][0]*shapes[layer][1])]
       #for j in range(W.size):
       #  W[j] = np.random.uniform(-bound, bound)
       W = np.array(W).reshape((shapes[layer][0], shapes[layer][1]))
       initialWeights.append(W)

    initialBiases = [gp.garray(0*np.random.rand(1, layerSizes[i])) for i in range(1, len(layerSizes))]
    initialWeights = [gp.garray(initWeightMatrix(shapes[i], scales[i], fanOuts[i], uniforms[i])) for i in range(len(shapes))]
        
    net = dbn_neuralnet_gpu(initialWeights, initialBiases, hidden_activation_function=hidden_activation_function, binary_visible=binary_visible)
    return net

    
def demo_xor():
    '''Demonstration of backprop with classic XOR example
    '''
    data = np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
    targets = np.array([[0.],[1.],[1.],[0.]])

    layerSizes=[2,2,1]
    scales = [0.05 for i in range(len(layerSizes)-1)]
    fanOuts = [None for i in range(len(layerSizes)-1)]
    learn_rate = 0.1
    epochs = 30

    nn = buildNet(layerSizes=layerSizes, scales=scales,fanOuts=fanOuts, binary_visible=True, hidden_activation_function=Sigmoid())
    
    print "initial parameters"
    print "=================="
    print "W 1", nn.weights[0].shape
    print nn.weights[0]
    print "bias 1", nn.biases[0].shape
    print nn.biases[0]
    print "W 2", nn.weights[1].shape
    print nn.weights[1]
    print "bias 2", nn.biases[1]
    print nn.biases[1]
    print "=================="
    
    #mbStream = (sampleMinibatch(2, data, targets) for unused in itertools.repeat(None))
    minibatch_stream = gpu_buffer(data=data,label=targets, batch_size = 2, randomize = True, gpu_buffer_size_MB=200).iter_minibatch()
    
    learn_rate = 0.1
    epochs = 10
    mbPerEpoch = 2
    num_hidden_layers = len(layerSizes)-1
    learn_rates = [learn_rate for i in range(num_hidden_layers)]
    L2_costs = [0 for i in range(num_hidden_layers)]
    dropouts = [0 for i in range(num_hidden_layers)]

    start_time = time.time()
    for ep, (trCE, trEr, seconds) in enumerate(nn.fine_tune(minibatch_stream, epochs, mbPerEpoch, loss=None, learn_rates=learn_rates, momentum = 0.9, L2_costs=L2_costs,  dropouts=dropouts, output_activation_function=Sigmoid())):
        print "Epoch %d\tError %g\tLoss %g\tSeconds %g" % (ep, trCE, trEr, seconds)
    print "Total fine tuning seconds: %g" % (time.time()-start_time)

    print "network test:"
    minibatch_stream = gpu_buffer(data=data,label=targets, batch_size = 2, randomize = False, gpu_buffer_size_MB=200).iter_minibatch()

    output = nn.predict(minibatch_stream,asNumpy = True)
    print output.next()
    
    out = nn.cpu_model().predict(data)
    print out

if __name__ == "__main__":
    demo_xor()
