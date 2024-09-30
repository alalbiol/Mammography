import numpy as np
import matplotlib.pyplot as plt
import pathlib
from typing import List, Dict
from PIL import Image
from tqdm import tqdm
from annotation_utils import read_annotation_image
import joblib

SHORTES_SIZE = 1696 # both multiple of 32 and appox like dezso 1700x2100
LONGEST_SIZE = 2112 

def get_case_from_png(fn):    
    case_folder_in = pathlib.Path(fn).parent
    case = "/".join(case_folder_in.parts[-3:])
    return case

def abnormality_file_from_png(fn):
    #replace the .png with .OVERLAY in pathlib.Path
    
    return fn.with_suffix('.OVERLAY')
    

class AnnotatedImage(object):
    """  Class to manage the annotations of a image.
    """
    def __init__(self, image_file) -> None:
        self.image_file = image_file
        self.orientation = image_file.name.split('.')[1]
        self.abnormalities = self.read_annotations()
        self.img = None
        
    
    def read_annotations(self) -> List[Dict]:
        #read the overlay file
        abnormality_file = abnormality_file_from_png(self.image_file)
        if not abnormality_file.exists():
            # print("Warning: No annotation file found for ", self.image_file)
            # print("     " , abnormality_file)
            # print("PNG exists: ", self.image_file.exists())
            # print("OVERLAY exists: ", abnormality_file.exists())
            return  []
            
        return  read_annotation_image(abnormality_file)
    
    def read_image(self):
        img = cv2.imread(str(self.image_file))
        self.img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
    def get_im_size(self):
        if self.img is None:
            self.read_image()
        print("Image shape: ", self.img.shape)
        return  self.img.shape[1], self.img.shape[0]
    
    def show(self, ax=None):
        if self.img is None:
            self.read_image()
        
        if ax is not None:
            ax.imshow(self.img)
            ax.set_title(self.orientation)
            
            ax.set_xlim(0, self.img.shape[1])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.margins(x=0)
            
            for abnormality in self.abnormalities:
                outline = abnormality['outline']
                ax.fill(outline[0], outline[1], 'r', alpha=0.3)
                #now plot the bounding box
                bb = abnormality['bounding_box']
                xmin = bb[0]
                ymin = bb[1]
                width = bb[6]
                height = bb[7]
                
                ax.add_patch(
                    plt.Rectangle((xmin, ymin),
                                width,
                                height, fill=False,
                                edgecolor='red', linewidth=1.5))
                
                
            #ax.set_aspect('equal')
        else:
            plt.imshow(self.img)
            plt.title(self.orientation)
            plt.show()
        
    def __str__(self) -> str:
        return f"AnnotatedImage: {self.image_file.name} with {len(self.abnormalities)} abnormalities"
    

def get_abnormality_class(abnormality):
    if abnormality['pathology'] == 'MALIGNANT':
        return 1
    
    if 'BENIGN' in abnormality['pathology']:
        return 0
    
    if  abnormality['pathology']=='UNPROVEN':
        if abnormality['breast_malignant']:
            return 1
        else:
            return 0
    return 0
    
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
    

class DDSM(object):
    def __init__(self, dataset_folder) -> None:
        if isinstance(dataset_folder, str):
            dataset_folder = pathlib.Path(dataset_folder)
        
        self.dataset_folder = dataset_folder
        self.cases = self.get_cases()
    
        self.all_annotated_cases = {case: self.get_annotated_images(case) for case in self.cases}
    
    def get_cases(self):
        all_png_images = list(self.dataset_folder.glob('**/*.png'))
        print("Number of images: ", len(all_png_images))
        
        all_cases = list(set([get_case_from_png(fn) for fn in all_png_images]))
            
        
        print("Number of cases: ", len(all_cases))
        
        return all_cases
    
    def get_annotated_images(self, case):
        case_folder = self.dataset_folder / case
        png_images = list(case_folder.glob('*.png'))
        annotated_images = [AnnotatedImage(case_folder / fn) for fn in png_images]
        return annotated_images
    
    def print_num_abnormalities(self):
        for case in sorted(self.cases):
            print(f"{case}: {len(self.all_annotated_cases[case])} images")
            for k in range(len(self.all_annotated_cases[case])):
                print(f"    {self.all_annotated_cases[case][k].image_file.name.split('.')[1]}: {len(self.all_annotated_cases[case][k].abnormalities)} abnormalities")
    
    def __str__(self) -> str:
        return f"DDSM dataset with {len(self.cases)} cases"
    
    def show_case(self, case):
        annotated_images = self.all_annotated_cases[case]
        
        total_height = 0
        total_width = 0
        for k in range(len(annotated_images)):
            orientation = annotated_images[k].orientation
            w, h = annotated_images[k].get_im_size()
            if orientation in  ['LEFT_CC', 'RIGHT_CC']:
                total_height += h
                total_width += w
 
        res = 1200 # pixels per inch        
        num_case = case.split("/")[-1]
        print("fig size: ", total_width//res, total_height//res)
        fig, axs = plt.subplots(2,2, figsize=(total_width//res, total_height//res),layout="constrained")
        
        
        
        for k in range(len(annotated_images)):
            orientation = annotated_images[k].orientation
            
            if 'LEFT' in orientation:
                col = 1
            else:
                col = 0
            if 'CC' in orientation:
                row = 0
            else:
                row = 1
            
            annotated_images[k].show(ax = axs[row, col])
        
        
        
        fig.suptitle(f"{num_case}")
        fig.subplots_adjust(wspace=0, hspace=0)
        fig.tight_layout()
        
    def split_train_val(self, seed = 42, fraction = 0.8):
        np.random.seed(seed)
        all_cases = self.cases
        np.random.shuffle(all_cases)
        num_train = int(fraction * len(all_cases))
        train_cases = all_cases[:num_train]
        val_cases = all_cases[num_train:]
        
        return train_cases, val_cases
    
    def generate_Y8_dataset_case(self, case, images_folder, labels_folder):
        annotated_images = self.all_annotated_cases[case]
        for annotated_image in annotated_images:
            #load image using PIL.Image
            orig_img = Image.open(annotated_image.image_file)
            orig_img = np.array(orig_img)
            
            #print("Processing: ", annotated_image.image_file.name)

            img, scale_factor = process_image_for_yolo(orig_img)
            
            W, H = img.shape[1], img.shape[0]
            
            image_file = images_folder / annotated_image.image_file.name
            
            cv2.imwrite(str(image_file), img)
            
            label_file = labels_folder / annotated_image.image_file.with_suffix('.txt').name
            with open(label_file, 'w') as f:
                for abnormality in annotated_image.abnormalities:
                    object_class = get_abnormality_class(abnormality)
                    #[xmin, ymin, xmax, ymax, centerx, centery, width, height]
                    center_x = abnormality['bounding_box'][4] * scale_factor / W
                    center_y = abnormality['bounding_box'][5] * scale_factor / H
                    width = abnormality['bounding_box'][6] * scale_factor / W
                    height = abnormality['bounding_box'][7] * scale_factor/ H 
                
                    f.write(f"{object_class} {center_x} {center_y} {width} {height}\n")


    
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
    dataset_folder = pathlib.Path("/home/alalbiol/Data/mamo/DDSM_png")
    ddsm = DDSM(dataset_folder)
    
    ddsm.print_num_abnormalities()
    train_cases, val_cases = ddsm.split_train_val()
    ddsm.generate_Y8_dataset(train_cases, "/tmp/ddsm_yolo/training", parallel=True)
    ddsm.generate_Y8_dataset(val_cases, "/tmp/ddsm_yolo/validation", parallel=True)
    
    # debug_cases = [case for case in ddsm.cases if "case3161" in case]
    # print("Debug cases: ", debug_cases)    
    # ddsm.generate_Y8_dataset(debug_cases, "/tmp/ddsm_yolo/training", parallel=False)