import pathlib
import os
import cv2
import pandas as pd
from tqdm import tqdm
import albumentations as A
import matplotlib.pyplot as plt
import numpy as np



def read_annotations(annotations_file):
    annotations = []
    with open(annotations_file, 'r') as f:
        for line in f:
            fields = line.strip().split()
            annotations.append(fields)
    return annotations


def get_number_of_annotations(label_path):
    """
    Get the number of annotations in a label file
    """
    
    with open(label_path, 'r') as f:
        return len(f.readlines())


def visualize(image, bboxes, format='yolo'):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))  
    ax.imshow(image)
    img_width, img_height = image.shape[1], image.shape[0]
    if isinstance(bboxes, list):
        bboxes = np.array(bboxes)
    for k in range(bboxes.shape[0]):
        if format == 'yolo':
            xc, yc, w, h = bboxes[k]
            xc = xc * img_width
            yc = yc * img_height
            w = w * img_width
            h = h * img_height
            x = xc - w / 2
            y = yc - h / 2
        else:
            x, y, w, h = bboxes[k]
        rect = plt.Rectangle((x, y), w, h, fill=False, color='red')
        ax.add_patch(rect)
    plt.show()
    

def create_annotations_df(original_dataset_path, debug = False):
    original_dataset_path = pathlib.Path(original_dataset_path)
    
    crop_size = 640
    all_annotations = []
    for partition in ['training', 'validation']:
        orig_images_folder = original_dataset_path / partition / 'images'
        orig_labels_folder = original_dataset_path / partition / 'labels'
             
        num_discarded = 0
        num_selected = 0
        
        print(f"Processing {partition} partition: {orig_images_folder}")
        all_images = list(orig_images_folder.glob('*.png'))
        num_annotations_all = 0
        for k, image_path in tqdm(enumerate(all_images),  desc=f"Processing {partition} partition"): 
            
            if debug and k > 20:
                break
            
            label_path = orig_labels_folder / (image_path.stem + '.txt')
            
            
            
            if label_path.exists() and get_number_of_annotations(label_path) > 0:
                
                img = cv2.imread(str(image_path))
                im_width, im_height = img.shape[1], img.shape[0]
                annotations = read_annotations(label_path)
                
                num_annotations = len(annotations)
                num_annotations_all += num_annotations
                
                for annotation in annotations:
                    annotation_dict = {}
                    annotation_image = image_path
                    annotation_dict['partition'] = partition
                    annotation_dict['image_id'] = annotation_image.stem
                    annotation_dict['image'] = str(annotation_image)
                    annotation_dict['class'] = int(annotation[0])
                    annotation_dict['yolo_xcenter'] = float(annotation[1])
                    annotation_dict['yolo_ycenter'] = float(annotation[2])
                    annotation_dict['yolo_width'] = float(annotation[3])
                    annotation_dict['yolo_height'] = float(annotation[4])
                    annotation_dict['xcenter'] = int(float(annotation[1]) * im_width)
                    annotation_dict['ycenter'] = int(float(annotation[2]) * im_height)
                    annotation_dict['width'] = int(float(annotation[3]) * im_width)
                    annotation_dict['height'] = int(float(annotation[4]) * im_height)
                    annotation_dict['xmin'] = int(annotation_dict['xcenter'] - annotation_dict['width'] / 2)
                    annotation_dict['ymin'] = int(annotation_dict['ycenter'] - annotation_dict['height'] / 2)
                    annotation_dict['xmax'] = int(annotation_dict['xcenter'] + annotation_dict['width'] / 2)
                    annotation_dict['ymax'] = int(annotation_dict['ycenter'] + annotation_dict['height'] / 2)
                    
                    crop_xmin = max(0, annotation_dict['xcenter']-crop_size//2)
                    crop_xmax = min(im_width, annotation_dict['xcenter']+crop_size//2)
                    if crop_xmin == 0:
                        crop_xmax = crop_size
                    if crop_xmax == im_width:
                        crop_xmin = im_width - crop_size
                    crop_ymin = max(0, annotation_dict['ycenter']-crop_size//2)
                    crop_ymax = min(im_height, annotation_dict['ycenter']+crop_size//2)
                    if crop_ymin == 0:
                        crop_ymax = crop_size
                    if crop_ymax == im_height:
                        crop_ymin = im_height - crop_size
                    
                    
                    annotation_dict['crop_xmin'] = crop_xmin
                    annotation_dict['crop_ymin'] = crop_ymin
                    annotation_dict['crop_xmax'] = crop_xmax
                    annotation_dict['crop_ymax'] = crop_ymax
                    annotation_dict['num_annotations'] = num_annotations
                    

                    all_annotations.append(annotation_dict)
      
                num_selected += 1
            else:
                num_discarded += 1
            #update tqdm progress bar with number of discarded and selected images
            #tqdm.write(f"num_anot total  {num_annotations_all}")
            
             
                
        print(f"Discarded {num_discarded} images with no annotations")
        print(f"Selected {num_selected} images with annotations")
        
    all_annotations_df = pd.DataFrame(all_annotations)
    print("Number of annotations: ", len(all_annotations_df), " ", num_annotations_all)
    return all_annotations_df    
        

def fix_bboxes(bboxes, id, format='yolo'):
    if format == 'yolo':
        bboxes_xmin = bboxes[:, 0] - bboxes[:, 2] / 2
        bboxes_ymin = bboxes[:, 1] - bboxes[:, 3] / 2
        bboxes_xmax = bboxes[:, 0] + bboxes[:, 2] / 2
        bboxes_ymax = bboxes[:, 1] + bboxes[:, 3] / 2
        
        if np.any(bboxes_xmin < 0) or np.any(bboxes_ymin < 0) or np.any(bboxes_xmax > 1) or np.any(bboxes_ymax > 1):
            print("Bboxes out of range for image:", id)
            print(bboxes_xmin, bboxes_ymin, bboxes_xmax, bboxes_ymax)
            print(bboxes)
            bboxes_xmin = np.clip    (bboxes_xmin, 0, 1)
            bboxes_ymin = np.clip(bboxes_ymin, 0, 1)
            bboxes_xmax = np.clip(bboxes_xmax, 0, 1)
            bboxes_ymax = np.clip(bboxes_ymax, 0, 1)
            bboxes[:, 0] = (bboxes_xmin + bboxes_xmax) / 2
            bboxes[:, 1] = (bboxes_ymin + bboxes_ymax) / 2
            bboxes[:, 2] = bboxes_xmax - bboxes_xmin
            bboxes[:, 3] = bboxes_ymax - bboxes_ymin
        
    return bboxes
      
def extract_crop(annotation,df):
    """ Given an annotation extracts the crop, search for all 
    annotations in the same image and returns the annotations and the image"""
        
    id = annotation['image_id']
    
    annot_image = df.loc[df['image_id'] == id]
    
    bboxes = annot_image[['yolo_xcenter',     'yolo_ycenter', 'yolo_width', 'yolo_height']].values

    fix_bboxes(bboxes, id)

    labels = annot_image['class'].values
    
    img = cv2.imread(annotation['image'])
    
    transform = A.Compose([
        A.Crop(x_min=annotation['crop_xmin'], y_min=annotation['crop_ymin'], x_max=annotation['crop_xmax'], y_max=annotation['crop_ymax']),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['labels'], min_visibility=0.1))
    
    img_out = transform(image=img, bboxes=bboxes, labels=labels)
    
    return img_out
    


if __name__ == "__main__":
    
    debug = False
    
    original_dataset_path = "/home/alalbiol/Data/mamo/ddsm_yolo"
    dest_dataset_path = "/home/alalbiol/Data/mamo/ddsm_yolo_patches"

    
    original_dataset_path = pathlib.Path(original_dataset_path)
    dest_dataset_path = pathlib.Path(dest_dataset_path)
    
    all_annotations_df = create_annotations_df(original_dataset_path, debug = debug)
    
    annotation_image_num = {}
    
    for partition in ['training', 'validation']:
        orig_images_folder = original_dataset_path / partition / 'images'
        orig_labels_folder = original_dataset_path / partition / 'labels'
    
        dest_images_folder = dest_dataset_path / partition / 'images'
        dest_labels_folder = dest_dataset_path / partition / 'labels'
        
        pathlib.Path(dest_images_folder).mkdir(parents=True, exist_ok=True)
        pathlib.Path(dest_labels_folder).mkdir(parents=True, exist_ok=True)
        
        annotations_partition = all_annotations_df.loc[all_annotations_df['partition'] == partition]    
        print("Number of annotations in partition: ", len(annotations_partition))
        for k, annotation in tqdm(annotations_partition.iterrows()):
            img_out = extract_crop(annotation, all_annotations_df)
            
            
            
            img = img_out['image']
            bboxes = img_out['bboxes']
            labels = img_out['labels']
            
            patch_num = annotation_image_num.get(annotation['image_id'], 0)
            annotation_image_num[annotation['image_id']] = patch_num + 1

    
            annotation_image_name = pathlib.Path(annotation['image']).stem
            cv2.imwrite(str(dest_images_folder / f"{annotation_image_name}_{patch_num}.png"), img)
            with open(dest_labels_folder / f"{annotation_image_name}_{patch_num}.txt", 'w') as f:
                for k in range(len(bboxes)):
                    f.write(f"{labels[k]} {bboxes[k][0]} {bboxes[k][1]} {bboxes[k][2]} {bboxes[k][3]}\n")
        

 
                         
                         
