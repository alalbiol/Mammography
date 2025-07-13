#!/bin/bash

apptainer shell --nv -B /home/alalbiol/Data/mamo/CBIS-DDSM-segmentation/images:/inferenceData -B ../faster_rcnn_vgg16:/my_code caffe_cpu.sif
