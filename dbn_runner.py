import sys
import platform
import numpy as np
from common import *
from activation import *
from dbn_neuralnet import *

def mean_absolute_error(predictions,targets):
    #valid = data_io.get_valid_df()
    #valid["SalaryNormalized"]
    error = np.mean(np.absolute(predictions.flatten() - targets))
    return error 

def load_data(train_file,train_label_file,test_file,test_label_file):
    #print("Reading in the training data")
    #train = data_io.get_train_df()
    #sys.stdout.flush()
    print("Lading data and labels")
    train = np.load(train_file)
    train_labels = np.load(train_label_file)#.reshape(-1,1)
    test = np.load(test_file)
    test_labels = np.load(test_label_file)#[:np.newaxis]
    return train,train_labels,test,test_labels

def dbn_nn_only(train_data,train_labels,test_data,test_labels=None, binary_visible = True, batch_size=64,layer_sizes=[512, 512, 1],hidden_activation_function=Sigmoid(), nesterov=False,learn_rate_decay_half_life=20,\
                output_activation_function = Sigmoid(), loss_function = SquareLoss(), learn_rate=0.1,rbm_epochs=None, epochs=10, L2_cost=0.001, apply_L2cost_after=10, dropout = 0.5, gpu_buffer_size_MB = 256, cpu_buffer_size_MB = 4096,\
                validation_func=None, validation_frequency = 5, model_file="model.npz"):
    '''Deep neural net without pretrain'''
    layer_sizes.insert(0, train_data.shape[1])
       
    layerSizes = layer_sizes
    scales = [0.05 for i in range(len(layerSizes)-1)]
    fanOuts = [None for i in range(len(layerSizes)-1)]
    learn_rates = [learn_rate for i in range(len(layerSizes)-1)]
    L2_costs = [L2_cost for i in range(len(layerSizes)-1)]
    #epochs = epochs
    dropouts = [dropout for i in range(len(layerSizes)-1)]
    if(dropouts[0]>0.2): dropouts[0] = 0.2 #visible inputs get less dropout
    
    net = buildNet(layerSizes=layerSizes, scales=scales,fanOuts=fanOuts,  binary_visible=binary_visible, hidden_activation_function=hidden_activation_function)
    
    mbPerEpoch = int(np.ceil(1.0 * train_data.shape[0]/batch_size))
    
    minibatch_train = cpu_gpu_buffer(data=train_data,label=train_labels, batch_size = batch_size, randomize = True, gpu_buffer_size_MB=gpu_buffer_size_MB,cpu_buffer_size_MB=cpu_buffer_size_MB).iter_minibatch()    
    #minibatch_train = gpu_buffer(data=train_data[...],label=train_labels, batch_size = batch_size, randomize = True, gpu_buffer_size_MB=gpu_buffer_size_MB).iter_minibatch()
    
    start_time = time.time()
    for ep, (trCE, trEr, seconds) in enumerate(net.fine_tune(minibatch_train, epochs, mbPerEpoch, loss=validation_func, learn_rates=learn_rates, momentum = 0.9,learn_rate_decay_half_life=learn_rate_decay_half_life, \
                                                             L2_costs=L2_costs, apply_L2cost_after=apply_L2cost_after, dropouts=dropouts, output_activation_function=output_activation_function, loss_function=loss_function, nesterov=nesterov)):
        print "Epoch %d\tError %g\tLoss %g\tSeconds %g" % (ep, trCE, trEr, seconds)
        if(ep%validation_frequency == 0): #do a prediction every few epochs so that we don't waste cycles               
            minibatch_test = cpu_gpu_buffer(data=test_data,label=test_labels, batch_size = batch_size, randomize = False, gpu_buffer_size_MB=gpu_buffer_size_MB,cpu_buffer_size_MB=cpu_buffer_size_MB).iter_minibatch()    
            #minibatch_test = gpu_buffer(data=test_data[...],label=test_labels, batch_size = batch_size, randomize = False, gpu_buffer_size_MB=gpu_buffer_size_MB).iter_minibatch() 
            predicted = tuple(net.predict(minibatch_test,asNumpy=True))
            yield np.vstack(predicted), net
            
    print "fine tuning seconds: %g" % (time.time()-start_time)
    net.saveWeights(model_file)
    minibatch_test = cpu_gpu_buffer(data=test_data,label=test_labels, batch_size = batch_size, randomize = False, gpu_buffer_size_MB=gpu_buffer_size_MB,cpu_buffer_size_MB=cpu_buffer_size_MB).iter_minibatch()
    #minibatch_test = gpu_buffer(data=test_data[...],label=test_labels, batch_size = batch_size, randomize = False, gpu_buffer_size_MB=gpu_buffer_size_MB).iter_minibatch()
    predicted = tuple(net.predict(minibatch_test,asNumpy=True))
    yield np.vstack(predicted),net

def dbn_finetune_pretrained(train_data,train_labels,test_data,test_labels=None, binary_visible = True, batch_size=64,layer_sizes=[512, 512, 1],hidden_activation_function=Sigmoid(), nesterov=False,learn_rate_decay_half_life=20,\
                output_activation_function = Sigmoid(), loss_function = SquareLoss(), learn_rate=0.1,rbm_epochs=None, epochs=10, L2_cost=0.001, apply_L2cost_after=10, dropout = 0.5, gpu_buffer_size_MB = 256, cpu_buffer_size_MB = 4096,\
                validation_func=None, validation_frequency = 5, model_file="model.npz"):    
    '''Fine tune all weights pretrained with RBM'''
    layer_sizes.insert(0, train_data.shape[1])
    layerSizes = layer_sizes
    
    num_hidden_layers = len(layerSizes)-1
    scales = [0.05 for i in range(num_hidden_layers)]
    fanOuts = [None for i in range(num_hidden_layers)]
    learn_rates = [learn_rate for i in range(num_hidden_layers)]
    batch_sizes = [batch_size for i in range(num_hidden_layers)]
    rbm_epochs = [rbm_epochs for i in range(num_hidden_layers)]
    
    momentums = [0.9 for i in range(num_hidden_layers)]
    L2_costs = [L2_cost for i in range(num_hidden_layers)]
    dropouts = [dropout for i in range(num_hidden_layers)]
    if(dropouts[0]>0.2): dropouts[0] = 0.2 #visible inputs get less dropout

        
    mbPerEpoch = int(np.ceil(1.0 * train_data.shape[0]/batch_size))

    net = buildNet(layerSizes=layerSizes, scales=scales,fanOuts=fanOuts, binary_visible=binary_visible,hidden_activation_function=hidden_activation_function)
    
    if(not binary_visible): learn_rates[0] *= 0.1
        
    net.unsupervised_pretrain(num_hidden_layers,train_data,batch_sizes=batch_sizes,epochs=rbm_epochs,learn_rates=learn_rates,momentums=momentums,L2_costs=L2_costs,dropouts=dropouts)
    
    #prepare to do fine-tuning of weights
    learn_rates = [rate * 0.5 for rate in learn_rates]
    
    minibatch_train = cpu_gpu_buffer(data=train_data,label=train_labels, batch_size = batch_size, randomize = True, gpu_buffer_size_MB=gpu_buffer_size_MB,cpu_buffer_size_MB=cpu_buffer_size_MB).iter_minibatch()    
    #minibatch_train = gpu_buffer(data=train_data[...],label=train_labels, batch_size = mbsz, randomize = True, gpu_buffer_size_MB=gpu_buffer_size_MB).iter_minibatch()
    
    start_time = time.time()
    for ep, (trCE, trEr, seconds) in enumerate(net.fine_tune(minibatch_train, epochs, mbPerEpoch, loss=validation_func, learn_rates=learn_rates, momentum = 0.9,learn_rate_decay_half_life=learn_rate_decay_half_life, \
                                                             L2_costs=L2_costs, apply_L2cost_after=apply_L2cost_after, dropouts=dropouts, output_activation_function=output_activation_function, loss_function=loss_function, nesterov=nesterov)):
        print "Epoch %d\tError %g\tLoss %g\tSeconds %g" % (ep, trCE, trEr, seconds)
        if(ep%validation_frequency == 0): #do a prediction every few epochs so that we don't waste cycles               
            minibatch_test = cpu_gpu_buffer(data=test_data,label=test_labels, batch_size = batch_size, randomize = False, gpu_buffer_size_MB=gpu_buffer_size_MB,cpu_buffer_size_MB=cpu_buffer_size_MB).iter_minibatch()    
            #minibatch_test = gpu_buffer(data=test_data[...],label=test_labels, batch_size = mbsz, randomize = False, gpu_buffer_size_MB=gpu_buffer_size_MB).iter_minibatch() 
            predicted = tuple(net.predict(minibatch_test,asNumpy=True))
            yield np.vstack(predicted), net
            
    print "fine tuning seconds: %g" % (time.time()-start_time)
    net.saveWeights(model_file)
    minibatch_test = cpu_gpu_buffer(data=test_data,label=test_labels, batch_size = batch_size, randomize = False, gpu_buffer_size_MB=gpu_buffer_size_MB,cpu_buffer_size_MB=cpu_buffer_size_MB).iter_minibatch()
    #minibatch_test = gpu_buffer(data=test_data[...],label=test_labels, batch_size = mbsz, randomize = False, gpu_buffer_size_MB=gpu_buffer_size_MB).iter_minibatch()
    predicted = tuple(net.predict(minibatch_test,asNumpy=True))
    yield np.vstack(predicted), net

def dbn_classify_pretrained(train_data,train_labels,test_data,test_labels, binary_visible = True, batch_size=64, layer_sizes=[400, 512, 512, 10],hidden_activation_function=Sigmoid(), learn_rate_decay_half_life=20,\
                            output_activation_function=Sigmoid(),learn_rate=0.1, L2_cost=0.001, dropout = 0.5, rbm_epochs=10, epochs=10, gpu_buffer_size_MB = 200, validation_frequency = 5, validation_func=None,model_file="model.npz"):
    '''Use pretrained features for classification, not fine tuning pretrained weights'''    
    layer_sizes.insert(0, train_data.shape[1])
    mbsz = batch_size
    layerSizes = layer_sizes
    
    num_hidden_layers = len(layerSizes)-1
    scales = [0.05 for i in range(num_hidden_layers)]
    fanOuts = [None for i in range(num_hidden_layers)]
    learn_rates = [learn_rate for i in range(num_hidden_layers)]
    batch_sizes = [batch_size for i in range(num_hidden_layers)]
    rbm_epochs = [rbm_epochs for i in range(num_hidden_layers)]
    epochs = [epochs for i in range(num_hidden_layers)]
    momentums = [0.9 for i in range(num_hidden_layers)]
    L2_costs = [L2_cost for i in range(num_hidden_layers)]
    dropouts = [dropout for i in range(num_hidden_layers)]
    if(dropouts[0]>0.2): dropouts[0] = 0.2 #visible inputs get less dropout
        
    mbPerEpoch = int(np.ceil(1.0 * train_data.shape[0]/mbsz))
    
    net = buildNet(layerSizes=layerSizes, scales=scales,fanOuts=fanOuts, binary_visible=binary_visible, hidden_activation_function=hidden_activation_function)
    
    if(not binary_visible): learn_rates[0] *= 0.1
    
    net.unsupervised_pretrain(num_hidden_layers,train_data,batch_sizes=batch_sizes,epochs=rbm_epochs,learn_rates=learn_rates,momentums=momentums,L2_costs=L2_costs,dropouts=dropouts)
            
    minibatch_stream = gpu_buffer(data=train_data,label=None, batch_size = mbsz, randomize = False, gpu_buffer_size_MB=gpu_buffer_size_MB).iter_minibatch()
    print("Feature extraction for training data")
    start_time = time.time() 
    outputs = tuple(net.predict(minibatch_stream,True))
    train_features = np.vstack(outputs)
    print("%s Features extracted with %g seconds" % (str(train_features.shape), time.time()-start_time))
          
    minibatch_stream = gpu_buffer(data=test_data,label=None, batch_size = mbsz, randomize = False, gpu_buffer_size_MB=gpu_buffer_size_MB).iter_minibatch()
    print("Feature extraction for testing data")
    start_time = time.time() 
    outputs = tuple(net.predict(minibatch_stream,True))
    test_features = np.vstack(outputs)
    print("%s Features extracted with %g seconds" % (str(test_features.shape), time.time()-start_time))
    
    net=None
    minibatch_stream=None
    gp.free_reuse_cache() #to free gpu memory held net as we only need features
    
    #Train MLP classifier
    mbsz = batch_size
    layerSizes = [train_features.shape[1], 128, 32, 1]
    scales = [0.05 for i in range(len(layerSizes)-1)]
    fanOuts = [None for i in range(len(layerSizes)-1)]
    learn_rate = 0.1
    epochs = epochs[0]
    mbPerEpoch = int(np.ceil(1.0*train_features.shape[0]/mbsz))
    
    #output_activation_function = Softmax()
    
    net = buildNet(layerSizes, scales,fanOuts,output_activation_function, False)

    
    minibatch_train = gpu_buffer(data=train_features,label=train_labels, batch_size = mbsz, randomize = True, gpu_buffer_size_MB=gpu_buffer_size_MB).iter_minibatch()
    
    if(learn_rate>0.05):
        learn_rate *= 0.1
        
    start_time = time.time()
    for ep, (trCE, trEr, seconds) in enumerate(net.fine_tune(minibatch_train, epochs, mbPerEpoch, loss=validation_func, learn_rates=learn_rates, momentum = 0.9, learn_rate_decay_half_life=learn_rate_decay_half_life,\
                                                             L2_costs=L2_costs, dropouts=dropouts, output_activation_function=output_activation_function)):
        print "Epocch %d\tError %g\tLoss %g\tSeconds %g" % (ep, trCE, trEr, seconds)
        if(ep%validation_frequency==0): #do a prediction every few epochs so that we don't waste cycles               
            minibatch_test = gpu_buffer(data=test_data,label=test_labels, batch_size = mbsz, randomize = False, gpu_buffer_size_MB=gpu_buffer_size_MB).iter_minibatch()    
            outputs = tuple(net.predict(minibatch_test,asNumpy=True))
            outputs = np.vstack(outputs)
            yield outputs, net        
    print "Total fine tuning seconds: %g" % (time.time()-start_time)
    
    net.saveWeights(model_file)
    
    minibatch_test = gpu_buffer(data=test_features,label=None, batch_size = mbsz, randomize = False, gpu_buffer_size_MB=gpu_buffer_size_MB).iter_minibatch()
    outputs = tuple(net.predict(minibatch_test,True))
    outputs = np.vstack(outputs)
    #print "Test error rate:", sum_absolute_error(test_labels, outputs) / float(test_data.shape[0])
    yield outputs, net

def main(train_data,train_labels,test_data,test_labels):
    import argparse
    
    parser = argparse.ArgumentParser(description='Run dbn train and predict', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument('--dbn_nn', action='store_true', default=False)
    group.add_argument('--dbn_finetune', action='store_true', default=False)
    group.add_argument('--dbn_classify', action='store_true', default=False)

    parser.add_argument('--hidden_activation_func', type=str, default='Sigmoid',choices = ['Sigmoid','ReLU','ReLU_cap','ReLU_smooth','ReLU_smooth_cap','Tanh','LeTanh'])#non-linear activation function
    parser.add_argument('--output_activation_func', type=str, default='Sigmoid',choices = ['Sigmoid','Softmax','Tanh','LeTanh','Linear'])#Linear means identity
    parser.add_argument('--loss_func', type=str, default='SquareLoss',choices = ['SquareLoss','CrossEntropyLoss','AbsoluteLoss','MultiClassLogLoss'])
    parser.add_argument('--binary_visible', action='store_true', default=False)
    parser.add_argument('--nesterov', action='store_true', default=False)
    parser.add_argument('--early_stop', action='store_true', default=False)

    parser.add_argument('--layer_sizes', nargs='+',type=int,required=True)
    parser.add_argument('--learn_rate', type=float,default=0.1)
    parser.add_argument('--learn_rate_decay_half_life', type=int, default=100)
    
    parser.add_argument('--L2_cost', type=float,default=0)
    parser.add_argument('--apply_L2cost_after', type=int, default=10)
    parser.add_argument('--dropout', type=float,default=0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--rbm_epochs', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--gpu_buffer_size_MB', type=int, default=256)
    parser.add_argument('--cpu_buffer_size_MB', type=int, default=4096)
    parser.add_argument('--validation_frequency', type=int, default=5)
    
    parser.add_argument('--model_file', type=str, default=None) 
      
    #parser.add_argument('--data_normalizer',type=str, default='None',choices = ['None','Gaussian','Uniform_Minus_One','Uniform_Zero_One'])

    parser.add_argument('--target_logarithm',action='store_true', default=False)
    
    parser.add_argument('--test_run',action='store_true', default=False)
    parser.add_argument('--validation_func',type=str, default='sum_absolute_error',choices = ['sum_absolute_error','num_mistakes'])
    
    #args = parser.parse_args()
    args, unknown = parser.parse_known_args()

    if(args.model_file is None):
        args.model_file = "dbn_model_file_%d.npz" % random.randint(0,20131122)

    print("Command line:\n\tpython %s" % " ".join(sys.argv) )
    
    print('Arguments:')
    
    for k,v in args.__dict__.items():
        print('\t%s %s' % (k, v))
    
    if(args.test_run):#test with first 100 rows
        #idx = np.random.randint(train_data.shape[0], size=(100,))
        train_data = train_data[0:100]
        train_labels=train_labels[0:100]
        
        #idx = np.random.randint(test_data.shape[0], size=(100,))
        test_data = test_data[0:100]
        test_labels = test_labels[0:100]
    
    mutex=None #lock the gpu 0 on win32 platform, so that other process won't interfere
    if platform.system() == 'Windows' or  platform.system().startswith('CYGWIN_NT'):
        import namedmutex
        mutex = namedmutex.NamedMutex('deeplink_gpu_mutex')
        logger.info('Waiting for mutex %s' % mutex.name)
        mutex.acquire()
        logger.info('Holding mutex %s' % mutex.name)
    #normalizer,un_normalizer = normalizer_name_map[args.data_normalizer]
    
    #train_data,mu1,sigma1 = normalizer(train_data)    
    #test_data,_,_ = normalizer(test_data,mu1,sigma1)
    
    if train_labels.ndim==1: train_labels = train_labels[:,np.newaxis]
    if test_labels.ndim==1: test_labels = test_labels[:,np.newaxis]
    
    if(args.target_logarithm):
        train_labels = np.log(train_labels)        
        #test_labels = np.log(test_labels)
        
    target_normalizer,target_unnormalizer = target_activation_normalizer_map[args.output_activation_func]
    
    normalized_train_labels,mu,sigma = target_normalizer(train_labels)
    #normalized_test_labels,_,_ = target_normalizer(test_labels,mu,sigma)
    
    logger.info("Data shapes - train %s train_labels %s test %s test_labels %s" %(train_data.shape,train_labels.shape,test_data.shape,test_labels.shape))
    
    positionalargs=[train_data,normalized_train_labels,test_data]#,normalized_test_labels]
    
    metric_func=metric_map[args.validation_func]
    
    kwargs=dict(
                batch_size=args.batch_size,
                layer_sizes=args.layer_sizes,
                binary_visible=args.binary_visible,
                learn_rate=args.learn_rate,
                learn_rate_decay_half_life=args.learn_rate_decay_half_life,
                L2_cost = args.L2_cost,
                apply_L2cost_after = args.apply_L2cost_after,
                dropout = args.dropout,
                rbm_epochs = args.rbm_epochs,
                epochs=args.epochs,
                validation_func = metric_func,
                hidden_activation_function = activation_name_map[args.hidden_activation_func],
                output_activation_function = activation_name_map[args.output_activation_func],
                loss_function = loss_name_map[args.loss_func],
                gpu_buffer_size_MB = args.gpu_buffer_size_MB,
                validation_frequency = args.validation_frequency,
                model_file = args.model_file
                )

    if(args.dbn_nn): out_iter = dbn_nn_only(*positionalargs,**kwargs)
    if(args.dbn_finetune): out_iter = dbn_finetune_pretrained(*positionalargs,**kwargs)
    if(args.dbn_classify): out_iter = dbn_classify_pretrained(*positionalargs,**kwargs)
    

    # early-stopping parameters
    best_test_loss = np.inf
    
    patience = 5 if args.early_stop else args.epochs/5  # look as this many rounds regardless, each round 5 epochs
    patience_increase = 2     # wait this much longer when a new best is
                              # found
    improvement_threshold = 0.998  # a relative improvement of this much is
                                    # considered significant    
    for iter, (predicted, net) in enumerate(out_iter):
        predicted = target_unnormalizer(predicted,mu,sigma)
        
        if(args.target_logarithm):
            predicted=np.exp(predicted)
                
        if(metric_func is not None):
            current_loss = metric_func(predicted,test_labels)
            if(current_loss<best_test_loss):                
                # improve patience if loss improvement is good enough
                if current_loss < best_test_loss * improvement_threshold and patience - patience_increase  < iter:
                    patience += patience_increase
                                    
                best_test_loss = current_loss
                sys.stdout.write("\tBest score so far")#:%g"%(best_test_loss/test_labels.shape[0]))
            if(patience <= iter):
                logger.info("\tNot seeing significant gain, early stopping.") 
                break
        
        yield predicted, test_labels, net
        
    #release win32 named mutex if any
    if mutex is not None: 
        logger.info('Releasing mutex %s' % mutex.name)
        mutex.release()
