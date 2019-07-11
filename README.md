C/C++ implementation for PCDN, SCDN and CDN metioned in the paper:

Parallelized Coordinate Descent Newton Method for Efficient L1-Regularized Minimization.
https://ieeexplore.ieee.org/abstract/document/8661743
https://arxiv.org/abs/1306.4080


## Installation
============

On Unix systems, type

$ make

to build the `train' and `infer'
programs. type

$ make clean

to clean the built files.

Run them without arguments to show the usages.

The software has been tested on Ubuntu 12.04 x86_64.

## `train' Usage
=============
Usage: train [options] training_file test_file [model_file_name]

options:

-a algorithm: set algorithm type (default 0)
        0 -- CDN
        1 -- Shotgun CDN (SCDN)
        2 -- Parallel CDN (PCDN)

-s solver type : set type of solver (default 0)
        0 -- L1-regularized logistic regression with bias term
        1 -- L1-regularized L2-loss support vector classification

-c cost : set the parameter C (default 1)

-e epsilon : set tolerance of termination criterion
        |f^S(w)|_1 <= eps*min(pos,neg)/l*|f^S(w0)|_1,
        where f^S(w) is the minimum-norm subgradient at w

-g g -n n : to generate the experimental results of CDN using a decreasing
            epsilon values = eps/g^i, for i = 0,1,...,n-1 (default g=1.0 n=1)

-q : quiet mode (no screen outputs)

training_file:
        training set file

test_file:
        test set file

model_file_name:
        model file name
        If you do not set model_file_name, it will be set as the result file nam
e following ".model"

## `infer' Usage
=============

Usage: infer  test_file model_file output_file

test_file:
        test set file

model_file_name:
        model file name

output_file:
        output file name


## Datasets Download
=================

Type

$ python ./gen_data.py

The script will defaultly download 1 data set (real-sim) from LIBSVM Data page.  If you want to download more datasets, edit the "data_dict" in 'gen_data.py' to indicate data sets for generation.
For those datasets, we do a 80/20 split for training and testing. It then stores *.train and *.test in the 'data' directory. Note that you need bunzip2, which is called by gen_data.py


## Set #bundle_size, #threads
==========================

Edit line 121-123 of src/train.cpp :

int g_pcdn_thread_num = 0;  //#threads for pcdn. default (set as 0): num_procs -1; otherwise, set as other positive integer
int g_bundle_size = 1250;   // bundle size  for pcdn
int g_scdn_thread_num = 8;   // #threads for scdn

then type

$ make

## The Log Files
=============

With each run, two log files will be stored in 'log/' directory, with the name indicating configuration of the specific experiment. For example,

'pcdn_threads_3_bundle_1250_s_0_c_4.0_eps_1e-3_real-sim'

'pcdn_threads_3_bundle_1250_s_0_c_4.0_eps_1e-3_real-sim_verbosity'

indicate: algorithm: pcdn, threads: 3, bundle size: 1250, slover: 0, C: 4.0, epsilon: 1e-3, dataset: real-sim.

The first log file stores the contents printed on the terminal, the second log file stores outputs of each iteration, which could be used to generate the experimental results.


## Example
========

real-sim.train and real-sim.test are put as example dataset on the project webpage:

real-sim

bundle size: 1250

L1-regularized logistic regression with bias term:

$./train -a 2 -s 0 -c 4.0  -e 1e-3   ./data/real-sim.train ./data/real-sim.test model_lrb

$./infer ./data/real-sim.test model_lrb out_lrb


L1-regularized L2-loss support vector classification:

$ ./train -a 2 -s 1 -c 1.0  -e 1e-3   ./data/real-sim.train ./data/real-sim.test model_svc

$ ./infer ./data/real-sim.test model_svc out_svc

## Copyright:

Copyright (2019) [Yatao (An) Bian <yatao.bian@gmail.com> | yataobian.com]. Please cite the above paper if you use this code in your work.
