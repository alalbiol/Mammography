import argparse
import os
import pathlib
from ultralytics import YOLO # Here we import YOLO 
import cv2
import numpy as np
import supervision as sv
from ultralytics.utils.ops import non_max_suppression
import torch
from tqdm import tqdm

def read_annotations(annotations_file):
    annotations = []
    with open(annotations_file, 'r') as f:
        for line in f:
            fields = line.strip().split()
            x_center = float(fields[1])
            y_center = float(fields[2])
            width = float(fields[3])
            height = float(fields[4])
            x1 = x_center - width/2
            y1 = y_center - height/2
            x2 = x_center + width/2
            y2 = y_center + height/2
            class_id = int(fields[0])
            anot = [x1, y1, x2, y2, class_id]
            annotations.append(anot)
    return np.array(annotations)




def evaluate_dataset(dataset_path, model_weights, model_name):
    dataset_path = pathlib.Path(dataset_path)
    
    images_folder = dataset_path / 'images'
    output_folder = dataset_path / model_name
    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)
    
    
    yolo_model = YOLO(model_weights)
    yolo_model.model.conf = 0.001
    
    
    
    all_images = list(images_folder.glob('*.png'))
    if len(all_images) == 0:
        print("No images found in the dataset folder:", images_folder)
    print(f"Evaluating dataset with {len(all_images)} images")
    
    all_predictions = []
    all_gt = []
    
    for ind, image_path in tqdm(enumerate(all_images)):
        labels_path = dataset_path / 'labels' / (image_path.stem + '.txt')
        predictions_path = output_folder / (image_path.stem + '.txt')
        gt = read_annotations(labels_path)
        all_gt.append(gt)
        
        image = cv2.imread(str(image_path)).astype(np.float32)
        
        #print("evaluating", image_path, labels_path)
        
        # tranform_im = image.transpose(2, 0, 1)
        # tranform_im = torch.from_numpy(tranform_im).cuda()
        # tranform_im /= 255.0 
        # tranform_im = torch.unsqueeze(tranform_im, 0)
            
        
        results = yolo_model( image, verbose=False)
        
        # iou_thres = 0.3
        # conf_thres = 0.001
        # classes = 2
        # agnostic_nms = False
        # max_det = 1000
        
        # boxes = results[0].boxes.data
        # results = non_max_suppression(boxes, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        
        
        
        # detections = sv.Detections.from_ultralytics(results[0])
        # bounding_box_annotator = sv.BoundingBoxAnnotator()
        # label_annotator = sv.LabelAnnotator()

        # labels = [
        #     yolo_model.model.names[class_id]
        #     for class_id
        #     in detections.class_id
        # ]
        
        # annotated_image = image

        # annotated_image = bounding_box_annotator.annotate(
        #      scene=image, detections=detections)
        
        # print("detections", detections)
        # annotated_image = label_annotator.annotate(
        #     scene=annotated_image, detections=detections, labels=labels)
        
        # w, h = annotated_image.shape[:2]
        # gt_boxes = np.column_stack([gt[:,0]*w, gt[:,1]*h, gt[:,2]*w, gt[:,3]*h])
        # gt_boxes  = sv.Detections(
        #     xyxy=gt_boxes,
        #     class_id=gt[:, 4].astype(int),)
        
        # print("gt_boxes", gt_boxes)
        # annotated_image = bounding_box_annotator.annotate(
        #     scene=annotated_image, detections=gt_boxes)

        # cv2.imwrite("annotated_image.png", annotated_image)
        
        
        predictions =  []
        #for result in results[0]:
        
        result = results[0]
        
        with open(predictions_path, 'w') as f:
            for box in result.boxes:
            
                coords = box.xyxyn.cpu().numpy()[0]
                class_id = box.cls.item()
                confidence = box.conf.item()
                
                
                xc = (coords[0] + coords[2])/2
                yc = (coords[1] + coords[3])/2
                w = coords[2] - coords[0]
                h = coords[3] - coords[1]
                
                f.write(f"{class_id} {xc} {yc} {w} {h} {confidence}\n")

                if class_id==1 and confidence >0:
                    print("Found example: ", predictions_path)
                     
                       
        
        
    print("Finished evaluating dataset")
    
    all_predictions = all_predictions
    
    return



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate test dataset with YOLO model.')
    parser.add_argument('--dataset_path',  type=str, help='Path to the original dataset')
    parser.add_argument('--model_weights', type=str, help='Path to trained weights')
   
    args = parser.parse_args()
    
    
    
    
    dataset_path = pathlib.Path(args.dataset_path)
    
    model_name = args.model_weights.split('/')[-3]
    
    print("Model Name:", model_name)
    
    images_folder = dataset_path / 'images'
    yolo_model = YOLO(args.model_weights)
    evaluate_dataset(args.dataset_path, args.model_weights, model_name)
    
    