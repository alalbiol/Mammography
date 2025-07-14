#!/bin/bash
net1=/my_code/scoring/vgg16_frcnn_cad.caffemodel
#net1=/models/vgg16_faster_rcnn_iter_40000.caffemodel
#net2=/models/vgg16_faster_rcnn_iter_35000.caffemodel
#mode="express"
mode="full"

#detect cancer
#time python src/infer1.py 0 $net1 $mode &
#time python src/infer1.py 1 $net2 $mode &
#wait

python ./infer.py 0 $net1 $mode

#aggregate
#python src/agg.py /scratch/0.tsv /scratch/1.tsv
