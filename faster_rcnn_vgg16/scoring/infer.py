#!/usr/bin/env python

"""
This script detects cancer in test images and writes predictions.

Author: Dezso Ribli
"""

import os,sys
os.chdir('/opt/gemfield/py-faster-rcnn/tools')
sys.path.append('/opt/gemfield/py-faster-rcnn/tools')
import pathlib

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
import numpy as np
import caffe
import cv2
#import pandas as pd


cfg.TEST.SCALES=(1700,) #change scales
cfg.TEST.MAX_SIZE=2100 #change scales
cfg.TEST.NMS = 0.1 #change nms as in the paper

#debug options
# cfg.TEST.RPN_NMS_THRESH = 1.1 #change rpn nms thresh
# cfg.TEST.RPN_POST_NMS_TOP_N = 12000 #change rpn post nms top n
# cfg.TEST.HAS_RPN = True #set that net generates region proposals
# cfg.TEST.MAX_SIZE = 2100 #set max size
# cfg.TEST.RPN_MIN_SIZE = 0 #set rpn min size
# cfg.TEST.RPN_PRE_NMS_TOP_N = 6000 #set rpn pre n
# cfg.TEST.RPN_FEAT_STRIDE = 16 #set rpn feat stride
# cfg.TEST.RPN_SCALES = (8, 16, 32) #set rpn scales
# cfg.TEST.RPN_RATIOS = (0.5, 1, 2) #set rpn ratios
# cfg.TEST.RPN_NMS_THRESH = 1.1 #set rpn nms thresh
# cfg.TEST.RPN_POST_NMS_TOP_N = 6000 #set rpn post n



def detect(pred_fn,d=None,meta_d='/metadata/',
           prototxt='/models/vgg16_faster_rcnn_test.prototxt',
           caffemodel='/models/vgg16_faster_rcnn.caffemodel',
           mode='full'):
    """Detect cancer in 'training' dataset."""
    #meta=pd.read_csv(meta_d+'images_crosswalk.tsv',sep='\t') #load meta
    with open(meta_d+'images_crosswalk.tsv', 'r') as f:
        lines = f.readlines()
    image_names = [line.strip() for line in lines[1:]]
    if mode == 'express': #for testing in express lane
        image_names=image_names[:1]
        
    print('Using ',len(image_names),' images') #print some info
    meta = {'filename': image_names}  # Create a dictionary to hold filenames
    
    net=load_net(prototxt,caffemodel) #load net
    meta['confidence'] = detect_ims(net,meta['filename']) #detect
    #save the tsv file
    #meta.to_csv(pred_fn,sep='\t',index=False)

def load_net(prototxt,caffemodel):
    """Load a faster rcnn net in test mode for prediction."""
    cfg.TEST.HAS_RPN = True  # set that net generates region proposals
    caffe.set_mode_gpu()
    caffe.set_device(GPU)
    cfg.GPU_ID = GPU  #set gpu in faster rcnn config
    net=caffe.Net(prototxt,caffe.TEST,weights=caffemodel)
    return net

def detect_ims(net,fn_list,d=None,cls_ind=2,offset=0):
    """Detect on a list of images."""
    confidence=[]
    
    print(fn_list)
    
    for i,fn in enumerate(fn_list):
        
        fn = d + fn if d is not None else fn  # prepend directory if needed
        im=load_im(fn) #load im
        scores,boxes=im_detect(net, im) #eval with net
        
        input_image = net.blobs['data'].data.copy()
        print("Input blob shape:", input_image.shape)
        print("Input image max value:", input_image.max())
        print("Input image min value:", input_image.min())
        print("Input image mean value:", input_image.mean(axis=(0,2,3)))
        
        activation = net.blobs['conv1_1'].data.copy()
        print("conv1_1 activation shape:", activation.shape)
        conv1_1_name = "/my_code/scoring/"+pathlib.Path(fn).stem+"_conv1_1.npy"
        np.save(conv1_1_name, activation)
        
        activation = net.blobs['conv2_2'].data.copy()
        print("conv2_2 activation shape:", activation.shape)
        conv2_2_name = "/my_code/scoring/"+pathlib.Path(fn).stem+"_conv2_2.npy"
        np.save(conv2_2_name, activation)

        activation = net.blobs['conv3_3'].data.copy()
        print("conv3_3 activation shape:", activation.shape)
        conv3_3_name = "/my_code/scoring/"+pathlib.Path(fn).stem+"_conv3_3.npy"
        np.save(conv3_3_name, activation)

        
        activation = net.blobs['conv4_3'].data.copy()
        print("conv4_3 activation shape:", activation.shape)
        conv4_3_name = "/my_code/scoring/"+pathlib.Path(fn).stem+"_conv4_3.npy"
        np.save(conv4_3_name, activation)

        
        activation = net.blobs['conv5_3'].data.copy()
        print("conv5_3 activation shape:", activation.shape)
        conv5_3_name = "/my_code/scoring/"+pathlib.Path(fn).stem+"_conv5_3.npy"
        np.save(conv5_3_name, activation)

        
        activation = net.blobs['rpn_cls_score'].data.copy()
        print("rpn_cls_score activation shape:", activation.shape)
        rpn_cls_score_name = "/my_code/scoring/"+pathlib.Path(fn).stem+"_rpn_cls_score.npy"
        np.save(rpn_cls_score_name, activation)


        activation = net.blobs['rpn_cls_score_reshape'].data.copy()
        print("rpn_cls_score_reshape activation shape:", activation.shape)
        rpn_cls_score_reshape_name = "/my_code/scoring/"+pathlib.Path(fn).stem+"_rpn_cls_score_reshape.npy"
        np.save(rpn_cls_score_reshape_name, activation)

        activation = net.blobs['rpn_cls_prob'].data.copy()
        print("rpn_cls_prob activation shape:", activation.shape)
        rpn_cls_prob_name = "/my_code/scoring/"+pathlib.Path(fn).stem+"_rpn_cls_prob.npy"
        np.save(rpn_cls_prob_name, activation)

        activation = net.blobs['rpn_cls_prob_reshape'].data.copy()
        print("rpn_cls_prob_reshape activation shape:", activation.shape)
        rpn_cls_prob_reshape_name = "/my_code/scoring/"+pathlib.Path(fn).stem+"_rpn_cls_prob_reshape.npy"
        np.save(rpn_cls_prob_reshape_name, activation)



        
        activation = net.blobs['rpn_bbox_pred'].data.copy()
        print("rpn_bbox_pred activation shape:", activation.shape)
        rpn_bbox_pred_name = "/my_code/scoring/"+pathlib.Path(fn).stem+"_rpn_bbox_pred.npy"
        np.save(rpn_bbox_pred_name, activation)

        activation = net.blobs['rois'].data.copy()
        print("rois activation shape:", activation.shape)
        rois_name = "/my_code/scoring/"+pathlib.Path(fn).stem+"_rois.npy"
        np.save(rois_name, activation)

        activation = net.blobs['pool5'].data.copy()
        print("pool5 activation shape:", activation.shape)
        pool5_name = "/my_code/scoring/"+pathlib.Path(fn).stem+"_pool5.npy"
        np.save(pool5_name, activation)

        activation = net.blobs['fc6'].data.copy()
        print("fc6 activation shape:", activation.shape)
        fc6_name = "/my_code/scoring/"+pathlib.Path(fn).stem+"_fc6.npy"
        np.save(fc6_name, activation)


        
        #print("All net blobs keys:", net.blobs.keys())
        
        
        confidence.append(scores[:,cls_ind].max()) #save the max score
        print("Image:",fn,"Confidence:",confidence[-1]) #print some info
        print("score shape:", scores.shape)  # print score shape
        #write_prog(100.*i/len(fn_list),offset) #write progress to txt
        for score, box in zip(scores, boxes):
            if score[cls_ind] > 0.5:
                print("   Detected class", cls_ind, " with score ", score[cls_ind])
    return confidence

def load_im(im_fn):
    """Load image."""
    
    #print("Loading image:", im_fn)  # print image filename
    parr=cv2.imread(im_fn, cv2.IMREAD_GRAYSCALE)  # load image in grayscale
    #print("Image shape:", parr.shape)  # print image shape
    #print("Image max before processing:", parr.max())  # print max value before processing
    
    parr=parr/parr.max() * 3.0
    parr=np.clip(parr,0.8,2.9)-0.8
    parr=(255.* parr/parr.max())
    #parr=parr/16.0 #make 0-256
    #make it an image and uint
    im=np.zeros((parr.shape[0],parr.shape[1])+(3,),dtype=np.uint8)
    im[:,:,0],im[:,:,1],im[:,:,2]=parr,parr,parr
    return im

def write_prog(perc,offset=0):
    """Write progress.txt"""
    with open('/progress.txt','w') as f:
        f.write(str(offset+perc))


if __name__=='__main__':
    GPU = int(sys.argv[1])
    caffemodel=sys.argv[2] #select model file
    mode = sys.argv[3] #express or full
    #pred_fn = '/scratch/'+str(GPU)+'.tsv'
    pred_fn = 'scores.tsv'  #output file
    detect(pred_fn,caffemodel=caffemodel,
           meta_d='/my_code/scoring/', 
           prototxt='/my_code/model/test.prototxt',
           mode = mode ) #detect and write output
