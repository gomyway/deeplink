deeplink
=======

Author: Leo Tang
(Profile: http://www.linkedin.com/in/lijuntang)


## Implementation of deep learning algorithms that works in practice ##

GPU-based python implementation of

1.  Feed-forward Neural Nets
2.  Restricted Boltzmann Machines
3.  Deep Belief Nets
4.  Smart gpu/cpu memory management reduces out of memory error
5.  Works on both linux and windows

Installation for windows
======================
1. install winpython 64-bit http://sourceforge.net/projects/winpython/ to a folder of your choice. (assuming it is c:\python27 for next steps)
2. install visual studio 2012 (visual studio 2012 express might work too, visual studio 2013 does not work with Nvidia cuda)
3. install Nvidia cuda toolkit 64-bit https://developer.nvidia.com/cuda-downloads 
4. unzip deeplink, compile cudamat for windows
  4.1  run 'developer command prompt for VS2012'
  4.2  go to folder deeplink/cudamat
  4.3  run 'nmake -f Makefile.win'
5. All set, to run the mnist example
	5.1 open command prompt 
	5.2 cd to folder deeplink/examples
	5.3 run command 'c:\Python27\python-2.7.5.amd64\python.exe mnist_runner.py --dbn_finetune --train_file ../data/mnist.h5 --output_activation_func Softmax --binary_visible --layer_sizes 512 512 10 --learn_rate 0.1 --batch_size 16 --rbm_epochs 30 --epochs 1000 --gpu_buffer_size_MB 256 --early_stop --hidden_activation_func Sigmoid'

for GPU memory optimization:
1. do not load all data into gpu at begining, instead, load a portion and do minibatch on that portion of data (todo) 
2. train RBM once at a time, save model and features, exit and run another process to train next (fresh gpu meory each time)

Built on top of the [cudamat](http://code.google.com/p/cudamat/) library by Vlad Mnih and gnumpy

Tips for gnumpy:

need both gpu_lock.py and run_on_me_or_pid_quit with executable permission

To test on cpu:
$ export GNUMPY_USE_GPU=no


to manually lock one gpu

/dev/shm

create symlic:
gpu_lock_0 -> /dev/null
gpu_lock_1 -> /dev/null



demos for mnist data:  http://yann.lecun.com/exdb/mnist/

 1. python dbn_neuralnet.py --mnist_nn --layer_sizes 784 512 512 10 --learn_rate 0.1 --batch_size 64 --epochs 10 --gpu_buffer_size_MB 200
 	2.4% error rate seconds: 219.395 on GTX 550 Ti
 2. python dbn_neuralnet.py --mnist_finetune --layer_sizes 784 512 512 10 --learn_rate 0.1 --batch_size 64 --epochs 10 --gpu_buffer_size_MB 50 > mnist_fine_tune.log 2>&1
 	1.44% error rate with seconds 641.89 on Quadro 2000 (10% slower than GTX 550 Ti)
 3. python dbn_neuralnet.py --mnist_finetune --layer_sizes 784 512 512 10 --learn_rate 0.1 --batch_size 64  --epochs 100 > 3-layer-dbn-100.log 2>&1
 	1.4% error rate with seconds 3080.52+240.852 = 3321.372 seconds on GTX 550 Ti
 4. python dbn_neuralnet.py --mnist_finetune --layer_sizes 784 512 512 10 --learn_rate 0.1 --batch_size 64 --epochs 100 --gpu_buffer_size_MB 100 > mnist_fine_tune-100.log 2>&1
 	1.34% error rate, 3103.61+221.416 seconds
 6. python mnist_runner.py  --dbn_nn --layer_sizes 2500 2000 1500 1000 500 10  --output_activation_func Softmax --learn_rate 0.1 --batch_size 64 --epochs 100 --gpu_buffer_size_MB 64 --train_file ~/deeplink/data/mnist.npz > mnist_dbn_6_layer_nn.log.`date +%Y-%m-%d-%H-%M`  2>&1 &
 	1.9% error rate,  9024.27 seconds
 

neuralnet class: http://www.willamette.edu/~gorr/classes/cs449/intro.html

convert hdf5
f2=h5py.File('mnist.h5','w')
f2.create_dataset('train_data',data=entity_features,dtype='f4', chunks=(8,entity_features.shape[1]), compression='lzf')
