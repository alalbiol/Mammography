
import pathlib
from PIL import Image
import numpy as np
import cv2
import torch
import pandas as pd
import os
from tqdm import tqdm
import pydicom
from pydicom import dcmread


from faster_rcnn_vgg16 import FasterRCNN, post_process_detections, post_process_detections2, load_caffe_weights_into_pytorch

DEBUG = False  # Set to True to enable debug mode

def load_im(filename, target_scale=1700, max_size=2100, dataset='DDSM'):
    """Load and preprocess image to match Faster R-CNN cfg.TEST scaling."""
    # Load and convert to float32

    if dataset == 'DDSM':
        image = np.array(Image.open(filename)).astype(np.float32)
        if DEBUG:
            print("image max before processing:", image.max())
        # DDSM-style normalization
        im = image / image.max() * 3.0
        image = np.clip(im, 0.8, 2.9) - 0.8
    elif dataset == 'INBREAST':
        image= dcmread(filename).pixel_array.astype(np.float32)
        
        # INBREAST-style normalization
        image = (image - 1000) / (2500 - 1000) * 255.0
    elif dataset == 'dream_pilot':
        image= dcmread(filename).pixel_array.astype(np.float32)
        image=(255.*image/image.max())
    
    else:
        # raise 
        raise ValueError(f"Unknown dataset: {dataset}. Supported datasets are 'DDSM' and 'INBREAST'.")
        
    image=(255.*image/image.max()) 
    image = np.clip(image, 0, 255)  
        

    # Compute scale factor based on shorter side
    im_shape = image.shape
    im_size_min = np.min(im_shape[:2])
    im_size_max = np.max(im_shape[:2])

    scale = float(target_scale) / float(im_size_min)
    # Prevent the longer side from exceeding max_size
    if np.round(scale * im_size_max) > max_size:
        scale = float(max_size) / float(im_size_max)

    # Resize image with computed scale
    image_resized = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    
    image_resized = np.stack((image_resized,) * 3, axis=-1)  # Convert grayscale to RGB by repeating channels
    pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]]).reshape((1, 1, 3))  # Mean pixel values for VGG16
    image_resized = image_resized - pixel_means  # Subtract mean pixel

    return image_resized, scale

def detect_lesions(filename, model, score_thresh = 0.05,  device='cuda', dataset='DDSM'):
    """
    Detects lesions in a mammography image using a Faster R-CNN model.

    Args:
        filename (str): Path to the image file.
        model (torch.nn.Module): Pre-trained Faster R-CNN model.

    Returns:
        list: A list of bounding boxes, labels, and scores for detected lesions.
              Each item in the list is a dict with keys: 'boxes', 'labels', 'scores'.
    """
    image, scale = load_im(filename, dataset=dataset)  # Load and preprocess image
    image = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) # to torch tensor, channel first and add batch dim
    
    if DEBUG:
        print("image max after processing:", image.max())
        print("image min after processing:", image.min())
        print("image mean after processing:", image.mean(axis=(0,2,3)))
        print("image shape after processing:", image.shape)
    
    
    image = image.to(device) # move to device
    
    im_info = torch.tensor([[image.shape[2], image.shape[3], scale]], device=device) # height, width, scale
    model.eval()  # Set the model to evaluation mode
    
    if DEBUG:
        activation = {}  # Dictionary to store activations
        
        def get_hook(layer_name):
            hook = lambda module, input, output: activation.update({layer_name: output.detach().cpu()})
            return hook
        
        def get_hook_2(layer_name):
            hook = lambda module, input, output: activation.update({layer_name: (output[0].detach().cpu(), output[1].detach().cpu())})
            return hook
            

        # Register hook to conv1_1
        hook = model.backbone.conv1_1.register_forward_hook(get_hook('conv1_1')) 
        hook = model.backbone.conv2_2.register_forward_hook(get_hook('conv2_2')) 
        hook = model.backbone.conv3_3.register_forward_hook(get_hook('conv3_3')) 
        hook = model.backbone.conv4_3.register_forward_hook(get_hook('conv4_3')) 
        hook = model.backbone.conv5_3.register_forward_hook(get_hook('conv5_3'))  # Register hook to conv5_3

        hook = model.rpn.rpn_cls_score.register_forward_hook(get_hook('rpn_cls_score'))  # Register hook to conv5_3
        hook = model.rpn.rpn_bbox_pred.register_forward_hook(get_hook('rpn_bbox_pred'))  # Register hook to conv5_3
        hook = model.proposal_layer.register_forward_hook(get_hook_2('rois_scores'))  # Register hook to conv5_3
        
        hook = model.rcnn_head.fc6.register_forward_hook(get_hook('fc6'))  # Register hook to conv5_3


    with torch.no_grad():
        _, _, cls_score, bbox_pred, rois = model(image, im_info)  # Forward pass
        
        if DEBUG:
            print("output of conv1_1.shape", activation['conv1_1'].shape)  # e.g. torch.Size([1, 64, 224, 224])
            np.save('conv1_1_activations',activation['conv1_1'].detach().cpu().numpy() )

            print("output of conv2_2.shape", activation['conv2_2'].shape)  # e.g. torch.Size([1, 64, 224, 224])
            np.save('conv2_2_activations',activation['conv2_2'].detach().cpu().numpy() )

            print("output of conv3_3.shape", activation['conv3_3'].shape)  # e.g. torch.Size([1, 64, 224, 224])
            np.save('conv3_3_activations',activation['conv3_3'].detach().cpu().numpy() )


            print("output of conv4_3.shape", activation['conv4_3'].shape)  # e.g. torch.Size([1, 64, 224, 224])
            np.save('conv4_3_activations',activation['conv4_3'].detach().cpu().numpy() )


            print("output of conv5_3.shape", activation['conv5_3'].shape)  # e.g. torch.Size([1, 64, 224, 224])
            np.save('conv5_3_activations',activation['conv5_3'].detach().cpu().numpy() )


            print("output rpn_cls_score.shape", activation['rpn_cls_score'].shape)  # e.g. torch.Size([1, 64, 224, 224])
            np.save('rpn_cls_score',activation['rpn_cls_score'].detach().cpu().numpy() )

            print("output rpn_bbox_pred.shape", activation['rpn_bbox_pred'].shape)  # e.g. torch.Size([1, 64, 224, 224])
            np.save('rpn_bbox_pred',activation['rpn_bbox_pred'].detach().cpu().numpy() )

            print("output rois.shape", activation['rois_scores'][0].shape)  # e.g. torch.Size([1, 64, 224, 224])
            print("output scores.shape", activation['rois_scores'][1].shape)  # e.g. torch.Size([1, 64, 224, 224])

            np.save('rois',activation['rois_scores'][0].detach().cpu().numpy() )

            
            print(f"cls_score shape: {cls_score.shape}, bbox_pred shape: {bbox_pred.shape}, rois shape: {rois.shape}")
            
            np.save('fc6',activation['fc6'].detach().cpu().numpy() )
        

        # Post-process to get detections (you might need to adapt this based on your model's output format)
        detections = post_process_detections2(cls_score, bbox_pred, rois, im_info, num_classes=3, 
                                             score_thresh=score_thresh,nms_thresh=0.1)
    return detections

def detect_lesions_in_files(filenames, model, score_thresh = 0.05,  device='cuda', dataset='DDSM'):
    """
    Detects lesions in a list of mammography image files.

    Args:
        filenames (list): A list of paths to image files.
        model (torch.nn.Module): Pre-trained Faster R-CNN model.

    Returns:
        list: A list of detections for each image. Each detection is a list of
              dictionaries with keys: 'boxes', 'labels', 'scores'.
    """
    all_detections = []
    for filename in tqdm(filenames, desc="Processing images"):
        tqdm.write(f"Processing {pathlib.Path(filename).name}")
        detections = detect_lesions(filename, model, score_thresh=score_thresh, device=device, dataset=dataset)
        all_detections.append(detections)
    return all_detections

def load_faster_rcnn_vgg16_model(model_path="../faster_rcnn_vgg16_weights", device='cuda'):
    """
    Load a pre-trained Faster R-CNN model with VGG16 backbone.

    Args:
        model_path (str): Path to the model file.
        device (str): Device to load the model on ('cuda' or 'cpu').

    Returns:
        torch.nn.Module: The loaded Faster R-CNN model.
    """
    #EXTRACTED_WEIGHTS_DIR = "../faster_rcnn_vgg16_weights" # Output directory from your extraction script



    # 2. Instantiate the PyTorch model
    NUM_CLASSES = 3 # From your Caffe prototxt: 'num_output: 3' for cls_score, and 'num_classes': 3 for roi-data

    # Choose between ROI Align and ROI Pooling
    # Set to True for ROI Align, False for ROI Pooling
    use_roi_align_option = False
    pytorch_model = FasterRCNN(num_classes=NUM_CLASSES, use_roi_align=use_roi_align_option)
    
    if os.path.exists(model_path):
        load_caffe_weights_into_pytorch(pytorch_model, model_path)
    else:
        print(f"Directory '{model_path}' not found. Cannot load extracted weights.")
        print("Please run the Caffe weight extraction script first.")

    
    pytorch_model.eval() # Set to evaluation mode for inference
    return pytorch_model

def main(model_path, file_input, score_thresh=0.3, outfile="detections.csv", dataset='DDSM'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_faster_rcnn_vgg16_model(model_path)
    model.to(device)

    if file_input.endswith('.csv'):
        df = pd.read_csv(file_input)
        filenames = df['filename'].tolist()
    else:
        filenames = [file_input]
        
    num_classes = 3  # Adjust based on your model's output classes

    all_detections = detect_lesions_in_files(filenames, model, 
                                             score_thresh=score_thresh, 
                                             device= device,
                                             dataset=dataset)

    
    with open(outfile, 'w') as f:
        f.write("id,box_x1,box_y1,box_x2,box_y2,score\n")
        for filename, detections in zip(filenames, all_detections):
            id = pathlib.Path(filename).stem  # Get the file name without extension
            print(f"Detections for {id}:")
            for c in range(1, num_classes):
                if len(detections[c]) > 0:
                    for detection in detections[c]:
                        box = detection[:4]
                        score = detection[4]
                        print(f"  Class {c}:   Box: [{box[0]:.2f}, {box[1]:.2f}, {box[2]:.2f}, {box[3]:.2f}], Score: {score:.4f}")
                        f.write(f"{id},{box[0]:.2f},{box[1]:.2f},{box[2]:.2f},{box[3]:.2f},{score:.4f}\n")
        
        
        # for i, detection in enumerate(detections): # iterates over images
        #     for c in range(num_classes-1):
        #         if len(detection[c]) > 0:
        #             print(f"  Instance {i + 1}:")
        #             for j in range(len(detection[c])):
        #                 box = detection[j][:4]
        #                 label = c + 1 
        #                 score = detection[j][4]
        #                 print(f"    Box: [{box[0]:.2f}, {box[1]:.2f}, {box[2]:.2f}, {box[3]:.2f}], Label: {label}, Score: {score:.4f}")
        #         else:
        #             print("  No lesions detected.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect lesions in mammography images.")
    parser.add_argument("--model_path", default = '../faster_rcnn_vgg16_weights', help="Path to the Faster R-CNN model file.")
    parser.add_argument("--file_input", help="Path to an image file or a CSV file containing a 'filename' column.")
    parser.add_argument("--score_thresh", type=float, default=0.05, help="Score threshold for detections.")
    parser.add_argument("--outfile", default="detections.csv", help="Output file to save detections.")
    parser.add_argument("--dataset", default='DDSM', choices=['DDSM', 'INBREAST', 'dream_pilot'], help="Dataset type for normalization.")
    args = parser.parse_args()

    main(args.model_path, args.file_input, score_thresh=args.score_thresh, 
         outfile=args.outfile, dataset=args.dataset)
