# 1. Import necessary libraries
from ultralytics import YOLO # Here we import YOLO 
import yaml                  # for yaml files 
import torch
from PIL import Image
import os
import cv2
import time

def train_yolo(dataset, model_name,  train_options, pretrained_weights=None):
    # 2. Choose our yaml file
   

    # 3. Create Yolo model
    if pretrained_weights is None:
        model = YOLO(model_name + '.yaml').load(model_name + '.pt')  # build from YAML and transfer weights
    else:
        print("Loading pretrained weights: ", pretrained_weights)
        model = YOLO(pretrained_weights)  # build from YAML and transfer weights
    # model = YOLO(model_name + '.yaml')
    # model = YOLO(model_name + '.yaml') # creates Yolo object from 'yolov8n.yaml' configuration file. 
    # model = YOLO(model_name + '.pt')   # Loads pretrained weights             

    # 4. Train the model

    model.train(data='{}'.format(dataset), **train_options) # train the model using the yaml file
    

if __name__ == "__main__":
    dataset = 'dataset.yaml' 
     
    train_options = {'imgsz': 2112, 
            'batch': -1, 
            'epochs': 1500, 
            'patience': 0, 
            'rect': False,
            'close_mosaic': 1500,
            'name':'yolov8n',
            'hsv_h': 0.0,
            'hsv_s': 0.0,
            'hsv_v': 0.0,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.1,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.5,
            'fliplr': 0.5,
            'mosaic': 0.0,
            'mixup': 0.0,
            'iou': 0.5,
            }
    #'yolov8x' '07-yolov8x-40-epochs'
    #'yolov8l',          '08-yolov8l-40-epochs',
    # 'yolov8m', '09-yolov8m-40-epochs',
    #'yolov8s', '10-yolov8s-40-epochs',
    #544
    imgszs = [640]
    
    first = 23
    for k, imgsz in enumerate(imgszs):
        num = first + k
        print("Starting training for size: ", imgsz)
        exp_name = f'{num:02d}-yolov8n-patch_{imgsz}'
        
        train_options['imgsz'] = imgsz
        train_options['name'] = exp_name
        model_name = 'yolov8n'
        #pretrained_weights = 'runs/detect/21-yolov8n-patch_640/weights/last.pt'
        pretrained_weights = None
        
        train_yolo(dataset, model_name, train_options, pretrained_weights)
     
     
