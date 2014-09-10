#!/bin/bash
export DEEPLINK_WAIT_GPU=yes

echo "Training and testing mnist dbn globally finetuning"

python mnist_runner.py --dbn_finetune --train_file ../data/mnist.h5 --output_activation_func Softmax --binary_visible --layer_sizes 512 512 10 --learn_rate 0.1 --batch_size 64 --rbm_epochs 30 --epochs 1000 --gpu_buffer_size_MB 256 --early_stop --hidden_activation_func ReLU > mnist_dbn_finetune_ReLU.log.`date +%Y-%m-%d-%H-%M`  2>&1 &

python mnist_runner.py --dbn_finetune --train_file ../data/mnist.h5 --output_activation_func Softmax --binary_visible --layer_sizes 512 512 10 --learn_rate 0.1 --batch_size 64 --rbm_epochs 30 --epochs 1000 --gpu_buffer_size_MB 256 --early_stop > mnist_dbn_finetune.log.`date +%Y-%m-%d-%H-%M`  2>&1 &


echo "Training and testing 6-layer mnist neuralnet"

python mnist_runner.py --dbn_nn --train_file ../data/mnist.h5 --output_activation_func Softmax --binary_visible --layer_sizes 512 512 10 --learn_rate 0.1 --batch_size 64 --epochs 1000 --gpu_buffer_size_MB 256 --early_stop > mnist_dbn_6_layer_nn.log.`date +%Y-%m-%d-%H-%M`  2>&1 &

python mnist_runner.py --dbn_nn --train_file ../data/mnist.h5 --output_activation_func Softmax --binary_visible --layer_sizes 512 512 10 --learn_rate 0.1 --batch_size 64 --epochs 1000 --gpu_buffer_size_MB 256 --hidden_activation_func ReLU --early_stop > mnist_dbn_6_layer_nn_ReLU.log.`date +%Y-%m-%d-%H-%M`  2>&1 &

#echo "Training and testing mnist nn classifier on dbn features"

#python mnist_runner.py --dbn_classify --train_file ../data/mnist.h5 --output_activation_func Softmax --binary_visible --layer_sizes 512 512 10 --learn_rate 0.1 --batch_size 64 --epochs 100 --gpu_buffer_size_MB 128 --dropout 0.5 --L2_cost 0.001 > mnist_dbn_6_layer_classify.log.`date +%Y-%m-%d-%H-%M`  2>&1 &
