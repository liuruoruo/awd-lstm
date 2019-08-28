#!/bin/bash

#CUDA_VISIBLE_DEVICES='0' mypython -u train_simple.py 1>simple/worker.log 2>&1 &
#CUDA_VISIBLE_DEVICES='1' mypython -u train_dynamic.py 1>dynamic/worker.log 2>&1 &
#CUDA_VISIBLE_DEVICES='2' mypython -u train_simple_2layer_share.py 1>simple_2layer_share/worker.log 2>&1 &
#CUDA_VISIBLE_DEVICES='3' mypython -u train_simple_4layer_share.py 1>simple_4layer_share/worker.log 2>&1 &
#CUDA_VISIBLE_DEVICES='4' mypython -u train_simple_ortho.py 1>simple_ortho/worker.log 2>&1 &
#CUDA_VISIBLE_DEVICES='5' mypython -u train_simple_vd_d.py 1>simple_vd_d/worker.log 2>&1 &
#CUDA_VISIBLE_DEVICES='6' mypython -u train_simple_yuyin.py 1>simple_yuyin/worker.log 2>&1 &
#CUDA_VISIBLE_DEVICES='7' mypython -u train_simple_exp.py 1>simple_exp/worker.log 2>&1 &


#CUDA_VISIBLE_DEVICES='0' mypython -u train_simple_exp1.py 1>simple_exp1/worker.log 2>&1 &
#CUDA_VISIBLE_DEVICES='1' mypython -u train_simple_exp2.py 1>simple_exp2/worker.log 2>&1 &
#CUDA_VISIBLE_DEVICES='2' mypython -u train_simple_exp3.py 1>simple_exp3/worker.log 2>&1 &
#CUDA_VISIBLE_DEVICES='3' mypython -u train_simple_exp4.py 1>simple_exp4/worker.log 2>&1 &
#CUDA_VISIBLE_DEVICES='4' mypython -u train_simple_exp5.py 1>simple_exp5/worker.log 2>&1 &
#CUDA_VISIBLE_DEVICES='5' mypython -u train_simple_exp6.py 1>simple_exp6/worker.log 2>&1 &
#CUDA_VISIBLE_DEVICES='6' mypython -u train_simple_exp7.py 1>simple_exp7/worker.log 2>&1 &
#CUDA_VISIBLE_DEVICES='7' mypython -u train_simple_exp8.py 1>simple_exp8/worker.log 2>&1 &


#CUDA_VISIBLE_DEVICES='2' mypython -u train_dynamic.py 1>dynamic/worker.log 2>&1 &
#CUDA_VISIBLE_DEVICES='3' mypython -u train_dynamic_share_vd.py 1>dynamic_share_vd/worker.log 2>&1 &
CUDA_VISIBLE_DEVICES='' mypython -u train_awd.py 1>awd/worker.log 2>&1 &
#CUDA_VISIBLE_DEVICES='1' mypython -u train_awd_exp.py 1>awd_exp/worker.log 2>&1 &
