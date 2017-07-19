#!/bin/bash
set -e

#python prepare.py
cd detector
eps=150
# CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --model res18 -b 12 --epochs $eps --save-dir res18 
maxeps=150
for (( i=25; i <= $maxeps; i+=5 ))
do
    echo "process $i epoch"
	
	if [ $i -lt 10 ]; then
	    CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --model res18 -b 16 --resume results/res18/00$i.ckpt --test 1
	elif [ $i -lt 100 ]; then 
	    CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --model res18 -b 16 --resume results/res18/0$i.ckpt --test 1
	elif [ $i -lt 1000 ]; then
	    CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --model res18 -b 16 --resume results/res18/$i.ckpt --test 1
	else
	    echo "Unhandled case"
    fi

    if [ ! -d "results/res18/baselinebboxlranchor/val$i/" ]; then
        mkdir results/res18/baselinebboxlranchor/val$i/
    fi
    mv results/res18/bbox/*.npy results/res18/baselinebboxlranchor/val$i/
done 
# cp results/res18/$eps.ckpt results/res18/baselinebbox/detector.ckpt
