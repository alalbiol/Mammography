import numpy as np
import matplotlib.pyplot as plt
import cv2
import pathlib
import pydicom
from typing import List, Dict
from PIL import Image
from tqdm import tqdm
import pandas as pd
from annotation_utils import read_annotation_image
import joblib

SHORTES_SIZE = 1696 # both multiple of 32 and appox like dezso 1700x2100
LONGEST_SIZE = 2112 

def get_case_from_image(fn):    
    case_folder_in = pathlib.Path(fn).parent
    case = "/".join(case_folder_in.parts[-3:])
    return case

    

class AnnotatedImageINbreast(object):
    """  Class to manage the annotations of a image.
    """
    def __init__(self, image_file, annotation_file = None) -> None:
        self.image_file = image_file
        self.annotation_file = annotation_file
        self.abnormalities = self.read_annotations() if annotation_file is not None else []
        self.id = self.get_id()
        self.img = None
        
    def __len__(self):
        return len(self.abnormalities)
        
    def get_id(self):
        return self.image_file.stem.split("_")[0]
    
    def read_annotations(self) -> List[Dict]:
        a=pd.read_csv(str(self.annotation_file),sep='\t')
        #create empty list[dict]
        annots=[]
        for i in range(len(a)):
            #is it benign or malignant? otherwise skip
            if a.desc[i][0] == 'b':
                patho='BENIGN'
            elif a.desc[i][0] == 'm':
                patho='MALIGNANT'
            else:
                continue
                
            #make xy from bounding box
            xmin=a.x0[i]
            xmax=a.x1[i]
            ymin=a.y0[i]
            ymax=a.y1[i]
            
            #load imshape
            annots.append({"label":patho,"xmin": xmin,"ymin": ymin, 
                           "xmax":xmax, "ymax": ymax})
        return annots
              
    def read_image(self):
        im = pydicom.dcmread(str(self.image_file)).pixel_array
        
        #there is no information how to transform the pixels
        # it looks like this range is good
        im=np.clip(im,1000,2500)-1000
        #rescale
        im=(255.*im/im.max()).astype(np.uint8)
        self.img = im
        return im
           
    def get_im_size(self):
        if self.img is None:
            self.read_image()
        print("Image shape: ", self.img.shape)
        return  self.img.shape[1], self.img.shape[0]
    
    def show(self, ax=None):
        if self.img is None:
            self.read_image()
            
        if ax is None:
            fig, ax = plt.subplots()   
                  
        ax.imshow(self.img, cmap='gray')
        ax.set_title(self.image_file.stem)
        
        ax.set_xlim(0, self.img.shape[1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.margins(x=0)
        
        for abnormality in self.abnormalities:
            xmin = abnormality['xmin']
            ymin = abnormality['ymin']
            width = abnormality['xmax'] - abnormality['xmin'] +1 
            height = abnormality['ymax'] - abnormality['ymin'] +1

            ax.add_patch(
                plt.Rectangle((xmin, ymin),
                            width,
                            height, fill=False,
                            edgecolor='red', linewidth=1.5))
             
    def __str__(self) -> str:
        return f"AnnotatedImage: {self.id} with {len(self.abnormalities)} abnormalities"
    


    
def calculate_scale_factor_dezzo(W,H, max_w = SHORTES_SIZE, max_h = LONGEST_SIZE ):
    
    scale_h = max_h / H
    scale_w = max_w / W
    
    scale = min(scale_h, scale_w)

    return scale     

def letterbox_image(image, max_w=SHORTES_SIZE, max_H=LONGEST_SIZE):
    '''resize image with unchanged aspect ratio using padding'''

    img_w, img_h = image.shape[1], image.shape[0]
    
    if img_w < max_w:
        pad_width = max_w - img_w
        image = np.pad(image, ((0, 0), (0, pad_width)), mode='constant')
    if img_h < max_H:
        pad_height = max_H - img_h
        image = np.pad(image, ((0, pad_height), (0, 0)), mode='constant')
    
    return image

def process_image_for_yolo(img):
    img = np.array(img)
    W, H = img.shape[1], img.shape[0]
    scale_factor = calculate_scale_factor_dezzo(W,H)
    img = cv2.resize(img, (0,0), fx=scale_factor, fy=scale_factor)
    img = letterbox_image(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img, scale_factor
    
def get_abnormality_class(abnormality):
    if abnormality == 'MALIGNANT':
        return 1
    
    if 'BENIGN' in abnormality:
        return 0
    
    return 0

class INbreast(object):
    def __init__(self, dataset_folder, annotations_folder) -> None:
        if isinstance(dataset_folder, str):
            dataset_folder = pathlib.Path(dataset_folder)
        if isinstance(annotations_folder, str):
            annotations_folder = pathlib.Path(annotations_folder)
        
        self.dataset_folder = dataset_folder
        self.annotations_folder = annotations_folder
        self.all_images = list ( (self.dataset_folder / 'AllDICOMs').glob('*.dcm') )
        self.all_images = sorted(self.all_images)
        self.all_annotations = list(pathlib.Path(self.annotations_folder).glob('*.tsv'))
        
        print("Number of images: ", len(self.all_images))
        print("Number of annotations: ", len(self.all_annotations))
    
        self.all_annotated_cases = {image: self.get_annotated_image(image) for image in self.all_images}
    
 
    
    def get_annotated_image(self, image):
        annotation_file = [fn for fn in self.all_annotations if fn.stem in image.name]
        
        annotation_file = annotation_file[0] if len(annotation_file) > 0 else None
        
        annotated_image = AnnotatedImageINbreast(image, annotation_file)
        return annotated_image
    
    def print_num_abnormalities(self):
        for image in sorted(self.all_images):
            print(f"  {str(self.all_annotated_cases[image])}")
    
    def __str__(self) -> str:
        return f"INbreast dataset with {len(self.all_images)} images"
    
    def show_case(self, image):
        annotated_image = self.all_annotated_cases[image]
        annotated_image.show()
    
    # def split_train_val(self, seed = 42, fraction = 0.8):
    #     np.random.seed(seed)
    #     all_cases = self.cases
    #     np.random.shuffle(all_cases)
    #     num_train = int(fraction * len(all_cases))
    #     train_cases = all_cases[:num_train]
    #     val_cases = all_cases[num_train:]
        
    #     return train_cases, val_cases
    
    def generate_Y8_dataset_case(self, case, images_folder, labels_folder):
        annotated_image = self.all_annotated_cases[case]
        
        orig_img = annotated_image.read_image()

        img, scale_factor = process_image_for_yolo(orig_img)
            
        W, H = img.shape[1], img.shape[0]
        
        image_file = images_folder / annotated_image.image_file.name
        image_file = image_file.with_suffix('.png')
        
        cv2.imwrite(str(image_file), img)
        
        label_file = labels_folder / annotated_image.image_file.with_suffix('.txt').name
        with open(label_file, 'w') as f:
            for abnormality in annotated_image.abnormalities:
                abnormality_class = abnormality['label']
                object_class = get_abnormality_class(abnormality_class)
                #[xmin, ymin, xmax, ymax, centerx, centery, width, height]
                xmin = abnormality['xmin'] 
                ymin = abnormality['ymin'] 
                xmax = abnormality['xmax']
                ymax = abnormality['ymax'] 
                centerx = (xmin + xmax) / 2 * scale_factor / W
                centery = (ymin + ymax) / 2 * scale_factor / H
                width = (xmax - xmin +1) * scale_factor / W
                height = (ymax - ymin +1) * scale_factor / H
            
                f.write(f"{object_class} {centerx} {centery} {width} {height}\n")


    
    def generate_Y8_dataset(self, cases, output_folder, parallel = False):
        output_folder = pathlib.Path(output_folder)
        if not output_folder.exists():
            output_folder.mkdir()
        
        images_folder = output_folder / "images"
        if not images_folder.exists():
            images_folder.mkdir()
            
        labels_folder = output_folder / "labels"
        if not labels_folder.exists():
            labels_folder.mkdir()    
        
        if parallel:
            from joblib import Parallel, delayed
            Parallel(n_jobs=-1)(delayed(self.generate_Y8_dataset_case)(case, images_folder, labels_folder) for case in tqdm(cases))
        else:
            for case in tqdm(cases):
                self.generate_Y8_dataset_case(case, images_folder, labels_folder)
                
            
            
        
        
    
    
    
if __name__ == "__main__":
    dataset_folder = pathlib.Path("/home/alalbiol/Data/mamo/INbreast")
    annotatiions_folder = pathlib.Path("./INbreast/my_rois")
    inbreast = INbreast(dataset_folder, annotatiions_folder)
    cases = inbreast.all_images
    
    dest_folder = "/tmp/inbreast_yolo/validation"
    if not pathlib.Path(dest_folder).exists():
        pathlib.Path(dest_folder).mkdir(parents=True, exist_ok=True)
    inbreast.generate_Y8_dataset(cases, dest_folder, parallel=True)
    
    
    # debug_cases = [case for case in ddsm.cases if "case3161" in case]
    # print("Debug cases: ", debug_cases)    
    # ddsm.generate_Y8_dataset(debug_cases, "/tmp/ddsm_yolo/training", parallel=False)