"""
==================================
 DIGITAL DATABASE FOR SCREENING MAMMOGRAPHY (DDSM) DATASET
==================================

Este modulo contiene las clases y funciones necesarias para cargar y procesar el dataset DDSM.
En particular, se incluyen las siguientes clases:
- BalancedPatchBatchSampler: Garantiza que cada lote contenga el mismo número de muesras de cada clase
- BalancedBatchSampler: Permite muestrear lotes de forma balanceada.
- RandomAffineTransform: Permite aplicar transformaciones aleatorias a las imagenes.
- IdentityTransform: Representa la transformacion identidad.
- PatchSampler: Permite muestrear parches de imagenes.
- DDSM_Patch_Dataset: Representa un conjunto de datos de parches de imagenes.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image
import random
import pandas as pd
import pathlib
import SimpleITK as sitk
import numpy as np
import cv2
import pytorch_lightning as pl
from utils.load_config import  get_parameter
import gzip
from utils.transforms import RandomContrast, RandomIntensity, Standardize
import albumentations as A
        
from utils.sample_patches_main import sample_positive_bb, sample_negative_bb,  sample_hard_negative_bb, sample_blob_negative_bb


class BalancedPatchBatchSampler(torch.utils.data.sampler.Sampler):
    """
    BalancedPatchBatchSampler ensures each batch contains an equal number of classes

    Args:
        dataset: A PyTorch Dataset object. Must have a `get_all_targets()` method returning labels.
        batch_size: Integer, the size of each batch. Must be even for balance.
    """
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        # recuperamos las etiquetas de las imagenes
        self.labels = dataset.get_all_targets()
        
        # identificamos las clases unicas (NORMAL, MASS_BENIGN, MASS_MALIGNANT, CALCIFICATION_BENIGN, CALCIFICATION_MALIGNANT)
        sample_classes = np.unique(self.labels)
        self.num_classes = len(sample_classes) # ¿cuántas hay en total? esto lo hacemos por si cambiamos el dataset, para que este mas generalizado
        
        print("Number of classes: ", self.num_classes)
        print("Batch size: ", batch_size)
        # nos aseguramos de que el tamaño del lote sea múltiplo del número de clases, así cada batch tendrá el mismo número de muestras de cada clase
        assert batch_size % self.num_classes == 0, "Batch size must be multiple of number of classes"

        # organizamos los índices de los datos según su clase
        # para cada i en sample_classes, obtenemos los índices donde la clase i aparece en self.labels (a esa misma clase)
        # ejemplo: self.labels = [0, 1, 0, 1, 0, 1, 0, 1] --> np.where(self.labels == 0)[0] → array([0, 2, 6])
        self.class_idx = [np.where(self.labels == i)[0] for i in sample_classes]   
        
        # tamaño del dataset
        self.dataset_size = len(self.labels) // self.num_classes * self.num_classes # make it multiple of num_classes
        

    def __iter__(self): # secuencia equilibrada de índices
        
        padded_class_idx = [] 
        num_samples_per_class = self.dataset_size // self.num_classes # número de muestras por clase
        for c in range(self.num_classes):
            shuffle_indices = np.random.permutation(self.class_idx[c]) # barajamos los índices de la clase c
            padded_class_idx.append(np.resize(shuffle_indices, num_samples_per_class)) # redimensionamos para asegurar num de muestras iguales por clase
        
        interleaved_idx = np.array(padded_class_idx).T # intercalamos los índices de las clases
        interleaved_idx = interleaved_idx.reshape(-1) 
        
        return iter(interleaved_idx)   
                    
      
                    
    def __len__(self):
        return self.dataset_size  # Total number of samples



class BalancedBatchSampler(torch.utils.data.sampler.Sampler):
    """
    BalancedBatchSampler ensures each batch contains an equal number of positive and negative samples.

    Args:
        dataset: A PyTorch Dataset object. Must have a `get_all_targets()` method returning labels.
        batch_size: Integer, the size of each batch. Must be even for balance.
    """
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.labels = dataset.get_all_targets()
        self.pos_indices = np.where(self.labels == True)[0]
        self.neg_indices = np.where(self.labels == False)[0]
        self.dataset_size = (len(self.labels) // 2) * 2

    def __iter__(self):
        num_pos_per_batch = self.batch_size // 2
        num_neg_per_batch = self.batch_size - num_pos_per_batch
        
        
        pos_indices = np.random.permutation(self.pos_indices)
        neg_indices = np.random.permutation(self.neg_indices)
        
        pos_indices = np.resize(pos_indices, self.dataset_size // 2) 
        neg_indices = np.resize(neg_indices, self.dataset_size // 2) 
        
        interleaved_indices = np.array([pos_indices, neg_indices]).T
        interleaved_indices = interleaved_indices.reshape(-1)
        
        return iter(interleaved_indices)
            
    def __len__(self):
        return self.dataset_size # Total number of samples

 

class RandomAffineTransform:
    def __init__(self, angle_range=(-180, 180), shear_range=(-0.1, 0.1), 
                 scale_range=(0.8, 1.2)):
        """
        Initialize the RandomAffineTransform with specified parameters.

        Parameters:
            angle_range (tuple): The range of angles (min, max) in degrees.
            shear_range (tuple): The range of shear values (min, max).
            scale_range (tuple): The range of scale values (min, max).
            center (tuple): The (x, y) center of rotation. If None, defaults to (0, 0).
        """
        self.angle_range = angle_range
        self.shear_range = shear_range
        self.scale_range = scale_range
        

    def generate(self, center):
        """
        Generates a random SimpleITK affine transformation based on the initialized parameters.

        Returns:
            SimpleITK.Transform: The generated affine transformation.
        """
        # Random angle in radians
        angle = random.uniform(self.angle_range[0], self.angle_range[1])
        
        # Random shear
        shear = random.uniform(self.shear_range[0], self.shear_range[1])
        
        
        # Random scale
        scale = random.uniform(self.scale_range[0], self.scale_range[1])

        # Create the affine transformation
        transform = sitk.AffineTransform(2)  # 2D affine transformation
        transform.SetCenter(center)
        
        # Apply rotation
        transform.Rotate(0, 1, angle)
        
        # Apply shear
        transform.Shear(0,1, shear)
        
        # Apply scaling
        transform.Scale([scale, scale])
        
        return transform
    
class IdentityTransform:
    def generate(self, center = None):
        return sitk.AffineTransform(2)

#patch policy:
# For normal images:
# 1. n random crops, m crops at blob positions
# For abnormal images:
# 1. n random positive crops, m crops at blob positions, n hard negative crops, m random negative crops
class PatchSampler(object):
    def __init__(self, patch_size: int, 
                 n_positive_crops=1, 
                 n_blob_crops=1, 
                 n_random_crops=1, 
                 n_hard_negative_crop=1, 
                 pos_cutoff = 0.80,
                 neg_cutoff = 0.1,
                 mirror = True,
                 affine_transform = None):
        
        self.patch_size = patch_size
        self.n_positive_crops = n_positive_crops
        self.n_blob_crops = n_blob_crops
        self.n_random_crops = n_random_crops
        self.n_hard_negative_crop = n_hard_negative_crop
        self.pos_cutoff = pos_cutoff
        self.neg_cutoff = neg_cutoff
        self.mirror = mirror
        
        if affine_transform is None:
            self.affine_transform = IdentityTransform()
        
        self.affine_transform = affine_transform
        self.blob_detector = self.create_blob_detector()
        
        
    def create_blob_detector(self):
        # Build a blob detector.
        blob_min_area=3 
        blob_min_int=.5
        blob_max_int=.85
        blob_th_step=10
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = blob_min_area
        params.maxArea = self.patch_size*self.patch_size
        params.filterByCircularity = False
        params.filterByColor = False
        params.filterByConvexity = False
        params.filterByInertia = False
        # blob detection only works with "uint8" images.
        params.minThreshold = int(blob_min_int*255)
        params.maxThreshold = int(blob_max_int*255)
        params.thresholdStep = blob_th_step
        # import pdb; pdb.set_trace()
        ver = (cv2.__version__).split('.')
        if int(ver[0]) < 3:
            blob_detector = cv2.SimpleBlobDetector(params)
        else:
            blob_detector = cv2.SimpleBlobDetector_create(params)
        
        return blob_detector
        
    def random_mirror(self, image, mask):
        if random.random() > 0.5:
            image = np.fliplr(image)
            if mask is not None:
                mask = np.fliplr(mask)
                
        if random.random() > 0.5:
            image = np.flipud(image)
            if mask is not None:
                mask = np.flipud(mask)
                
        return image, mask
        
    def sample_patches(self, image, label, mask=None, points = None):
        image_patches = []
        mask_patches = []
        labels = []
        
        n_random_crops = self.n_random_crops # can be increased if no blobs or hard negatives are found
        if label != 'NORMAL':
            for _ in range(self.n_positive_crops):
                image_patch, mask_patch = self.sample_abnormal_patch(image, mask, points)
                
                if self.mirror:
                    image_patch, mask_patch = self.random_mirror(image_patch, mask_patch)
                
                image_patches.append(image_patch)
                mask_patches.append(mask_patch)
                labels.append(label)
                
            for _ in range(self.n_hard_negative_crop):
                image_patch, mask_patch = self.sample_hard_negative_patch(image, mask,points)
                
                
                if self.mirror:
                    image_patch, mask_patch = self.random_mirror(image_patch, mask_patch)
                    
                if image_patch is None:
                    n_random_crops += (self.n_hard_negative_crop - k)
                    break
                    
                image_patches.append(image_patch)
                mask_patches.append(mask_patch)
                labels.append('NORMAL')
        else:
            n_random_crops += self.n_positive_crops + self.n_hard_negative_crop # so all batches have the same size
                
        for k in range(self.n_blob_crops):
            image_patch, mask_patch = self.sample_blob_patch(image, mask, points)
            
            if image_patch is None:
                n_random_crops += (self.n_blob_crops - k)
                #print("No blobs found, increasing random crops")
                break
            
            
            if self.mirror:
                image_patch, mask_patch = self.random_mirror(image_patch, mask_patch)

            image_patches.append(image_patch)
            mask_patches.append(mask_patch)
            labels.append('NORMAL')
    

        for _ in range(n_random_crops):
            image_patch, mask_patch = self.sample_random_patch(image, mask, points)
            if self.mirror:
                image_patch, mask_patch = self.random_mirror(image_patch, mask_patch)

            image_patches.append(image_patch)
            mask_patches.append(mask_patch)
            labels.append('NORMAL')

        return image_patches, labels, mask_patches
        
    def sample_abnormal_patch(self, image, mask, points):
        
        sitk_image = sitk.GetImageFromArray(image)
    
        # Get image center as rotation center
        width, height = image.shape[1], image.shape[0]
        center = [width / 2.0, height / 2.0]

        # Set up rotation transform
        
        transform = self.affine_transform.generate(center=center)
        
        inverse_transform = transform.GetInverse()
        
        transformed_points = np.array([inverse_transform.TransformPoint(point) for point in points])
        
        bbs = sample_positive_bb(transformed_points, self.patch_size, nb_abn=1, pos_cutoff=self.pos_cutoff )
    
        output_origin = [bbs[0][0]-bbs[0][2]/2, bbs[0][1]-bbs[0][3]/2]        
        
    
        # Apply transformation
        augmented_image = sitk.Resample(sitk_image, 
                                    size = [self.patch_size, self.patch_size],
                                    outputOrigin = output_origin,
                                    transform = transform,
                                    interpolator = sitk.sitkLinear, 
                                    defaultPixelValue = 0, 
                                    outputPixelType = sitk_image.GetPixelID())

        # Convert rotated SimpleITK image back to numpy array
        augmented_image_array = sitk.GetArrayFromImage(augmented_image)
        
        if mask is not None:
            sitk_mask = sitk.GetImageFromArray(mask)
            augmented_mask = sitk.Resample(sitk_mask, 
                                    size = [self.patch_size, self.patch_size],
                                    outputOrigin = output_origin,
                                    transform = transform,
                                    interpolator = sitk.sitkLinear, 
                                    defaultPixelValue = 0, 
                                    outputPixelType = sitk_mask.GetPixelID())
            augmented_mask_array = sitk.GetArrayFromImage(augmented_mask)
        else:
            augmented_mask_array = None
            
        
        return augmented_image_array, augmented_mask_array
               
    def sample_blob_patch(self, image, mask, points):
        
        key_pts = self.blob_detector.detect((image/image.max()*255).astype('uint8'))    
        #print("Number of keypoints found=", len(key_pts))

                
        sitk_image = sitk.GetImageFromArray(image)
    
        # Get image center as rotation center
        width, height = image.shape[1], image.shape[0]
        center = [width / 2.0, height / 2.0]

        # Set up rotation transform
        
        transform = self.affine_transform.generate(center=center)
        
        inverse_transform = transform.GetInverse()
        
        if points is None:   
            transformed_points = None    
        else:
            transformed_points = np.array([inverse_transform.TransformPoint(point) for point in points])
        
        transformed_key_pts = np.array([inverse_transform.TransformPoint([kp.pt[0], kp.pt[1]]) for kp in key_pts])
        
        bbs = sample_blob_negative_bb(transformed_key_pts, transformed_points, 
                                      self.patch_size, nb_bkg=1, neg_cutoff=self.neg_cutoff) 
        
        if len(bbs) == 0:
            return None, None # all blobs overlap with the abnormality
    
        output_origin = [bbs[0][0]-bbs[0][2]/2, bbs[0][1]-bbs[0][3]/2]        
        
    
        # Apply transformation
        augmented_image = sitk.Resample(sitk_image, 
                                    size = [self.patch_size, self.patch_size],
                                    outputOrigin = output_origin,
                                    transform = transform,
                                    interpolator = sitk.sitkLinear, 
                                    defaultPixelValue = 0, 
                                    outputPixelType = sitk_image.GetPixelID())

        # Convert rotated SimpleITK image back to numpy array
        augmented_image_array = sitk.GetArrayFromImage(augmented_image)
        
        if mask is not None:
            sitk_mask = sitk.GetImageFromArray(mask)
            augmented_mask = sitk.Resample(sitk_mask, 
                                    size = [self.patch_size, self.patch_size],
                                    outputOrigin = output_origin,
                                    transform = transform,
                                    interpolator = sitk.sitkLinear, 
                                    defaultPixelValue = 0, 
                                    outputPixelType = sitk_mask.GetPixelID())
            augmented_mask_array = sitk.GetArrayFromImage(augmented_mask)
        else:
            augmented_mask_array = None
            
        return augmented_image_array, augmented_mask_array

    def sample_random_patch(self, image, mask, points):
        sitk_image = sitk.GetImageFromArray(image)
    
        # Get image center as rotation center
        width, height = image.shape[1], image.shape[0]
        center = [width / 2.0, height / 2.0]

        # Set up rotation transform
        
        transform = self.affine_transform.generate(center=center)
        
        inverse_transform = transform.GetInverse()
        
        image_outline = [[0,0], [0, height], [width, height], [width, 0]]
        
        transformed_image_outline = np.array([inverse_transform.TransformPoint(point) for point in image_outline])
        
        if points is None:
            transformed_points = None
        else:
            transformed_points = np.array([inverse_transform.TransformPoint(point) for point in points])
        
        bbs = sample_negative_bb(transformed_image_outline, transformed_points, self.patch_size, nb_bkg=1, neg_cutoff=self.neg_cutoff )
    
        output_origin = [bbs[0][0]-bbs[0][2]/2, bbs[0][1]-bbs[0][3]/2]        
        
    
        # Apply transformation
        augmented_image = sitk.Resample(sitk_image, 
                                    size = [self.patch_size, self.patch_size],
                                    outputOrigin = output_origin,
                                    transform = transform,
                                    interpolator = sitk.sitkLinear, 
                                    defaultPixelValue = 0, 
                                    outputPixelType = sitk_image.GetPixelID())

        # Convert rotated SimpleITK image back to numpy array
        augmented_image_array = sitk.GetArrayFromImage(augmented_image)
        
        if mask is not None:
            sitk_mask = sitk.GetImageFromArray(mask)
            augmented_mask = sitk.Resample(sitk_mask, 
                                    size = [self.patch_size, self.patch_size],
                                    outputOrigin = output_origin,
                                    transform = transform,
                                    interpolator = sitk.sitkLinear, 
                                    defaultPixelValue = 0, 
                                    outputPixelType = sitk_mask.GetPixelID())
            augmented_mask_array = sitk.GetArrayFromImage(augmented_mask)
        else:
            augmented_mask_array = None
            
        
        return augmented_image_array, augmented_mask_array

    def sample_hard_negative_patch(self, image, mask, points):
        sitk_image = sitk.GetImageFromArray(image)
    
        # Get image center as rotation center
        width, height = image.shape[1], image.shape[0]
        center = [width / 2.0, height / 2.0]

        # Set up rotation transform
        
        transform = self.affine_transform.generate(center=center)
        
        inverse_transform = transform.GetInverse()
        
        if points is  None:
            transformed_points = None
        else:
            transformed_points = np.array([inverse_transform.TransformPoint(point) for point in points])
        
        bbs = sample_hard_negative_bb(transformed_points, self.patch_size, nb_bkg=1, neg_cutoff=self.neg_cutoff )
    
        output_origin = [bbs[0][0]-bbs[0][2]/2, bbs[0][1]-bbs[0][3]/2]        
        
    
        # Apply transformation
        augmented_image = sitk.Resample(sitk_image, 
                                    size = [self.patch_size, self.patch_size],
                                    outputOrigin = output_origin,
                                    transform = transform,
                                    interpolator = sitk.sitkLinear, 
                                    defaultPixelValue = 0, 
                                    outputPixelType = sitk_image.GetPixelID())

        # Convert rotated SimpleITK image back to numpy array
        augmented_image_array = sitk.GetArrayFromImage(augmented_image)
        
        if mask is not None:
            sitk_mask = sitk.GetImageFromArray(mask)
            augmented_mask = sitk.Resample(sitk_mask, 
                                    size = [self.patch_size, self.patch_size],
                                    outputOrigin = output_origin,
                                    transform = transform,
                                    interpolator = sitk.sitkLinear, 
                                    defaultPixelValue = 0, 
                                    outputPixelType = sitk_mask.GetPixelID())
            augmented_mask_array = sitk.GetArrayFromImage(augmented_mask)
        else:
            augmented_mask_array = None
            
        
        return augmented_image_array, augmented_mask_array



class DDSM_Patch_Dataset(Dataset):
    def __init__(self, split_csv, ddsm_annotations, root_dir, 
                 convert_to_rgb = True,
                 return_mask=False,
                 patch_sampler = None,
                 subset_size = None, 
                 include_normals = True,
                 normalize_input = False):
        
        self.split_csv = split_csv
        self.root_dir = pathlib.Path(root_dir)
        self.return_mask = return_mask
        self.patch_sampler = patch_sampler
        self.convert_to_rgb = convert_to_rgb
        self.include_normals = include_normals # include normal images in the dataset
        self.normalize_input = normalize_input
        
        self.ddsm_annotations = self.load_annotations(split_csv, ddsm_annotations)
        
        if subset_size is not None:
            print("Subsetting dataset to ", subset_size)
            self.ddsm_annotations = self.ddsm_annotations.sample(subset_size)
        
        #class_names = self.ddsm_annotations['label'].unique()
        class_names = ['NORMAL',
            'MASS_BENIGN',
            'CALCIFICATION_BENIGN',
            'MASS_MALIGNANT',
            'CALCIFICATION_MALIGNANT',
        ]
        self.class_to_idx = {class_name: i for i, class_name in enumerate(class_names)}
        self.idx_to_class = {i: class_name for i, class_name in enumerate(class_names)}
        
    
    def get_all_targets(self):
        """ Returns all the targets of the dataset, useful for sampling 

        Returns:
            np.array: with the targets of the dataset (per image)
        """
        return self.ddsm_annotations['label'].values
        

    def load_annotations(self, split_csv,  annotations_file):
        print("loading annotations")
        
        split_images = pd.read_csv(split_csv)
        
        
        if str(annotations_file).endswith('.json'):
            annotations = pd.read_json(annotations_file, orient='records', lines=True)
        else:
            with gzip.open(annotations_file, 'rt', encoding='utf-8') as f:
                annotations = pd.read_json(f, orient='records', lines=True)


        print("Number of annotations: ", len(annotations))
        
        
        # filter all annotations that are in the train_images
        annotations = annotations[annotations['image_id'].isin(split_images['ddsm_image'])]
        print("Number of annotations after filtering split: ", len(annotations))
        
        self.number_of_abnormal_annotations = len(annotations)
        print("Number of abnormal annotations: ", self.number_of_abnormal_annotations)
        
        if self.include_normals:
            # annotations only contains abnormal images, lets add normal images for training
            normals_images = [im for im in split_images['ddsm_image'].values if 'normals' in im]
            print(f"Including {len(normals_images)} normal images")
            
            normal_records = []
            for normal_image in normals_images:
                record = {
                    'type': "NORMAL",
                    'assesment': None,
                    'subtlety': None,
                    'pathology': "NORMAL",
                    'outline' : None,
                    'bounding_box': None,
                    'breast_malignant': False,
                    'image_id': normal_image,
                    'mask_id': None
                }
                normal_records.append(record)
                
            annotations = pd.concat([annotations, pd.DataFrame(normal_records)], ignore_index=True)
                
            print("Number of annotations after adding normals: ", len(annotations))
        else:
            print("Not including normal images")
        
        map_classes = {
            'NORMAL_NORMAL': 'NORMAL',
            'MASS_MALIGNANT': 'MASS_MALIGNANT',
            'MASS_BENIGN': 'MASS_BENIGN',
            'CALCIFICATION_MALIGNANT': 'CALCIFICATION_MALIGNANT',
            'CALCIFICATION_BENIGN': 'CALCIFICATION_BENIGN',
            'CALCIFICATION_BENIGN_WITHOUT_CALLBACK': 'CALCIFICATION_BENIGN',
            'MASS_BENIGN_WITHOUT_CALLBACK': 'MASS_BENIGN',
            'OTHER_BENIGN_WITHOUT_CALLBACK': 'OTHER_BENIGN',
            'MASS_UNPROVEN': 'MASS_UNPROVEN',
            'CALCIFICATION_UNPROVEN': 'CALCIFICATION_UNPROVEN',
            'OTHER_BENIGN': 'OTHER_BENIGN',
            }
        
        annotations['label'] = annotations['type'] + "_" + annotations['pathology']
        annotations['label'] = annotations['label'].map(map_classes)
        
        
        #unproven aparecen en cancer y benigno segun ddsm si aparecen con cancer en la mama es seguramente cancer y se extirpa
        #si es benigno seguramente no es nada

        # Dezso los etiquetaba todos UNPROVEN como benignos!

        for k, record in annotations.iterrows():
            if 'MASS_UNPROVEN' in record['label']:
                annotations.at[k, 'label'] = 'MASS_BENIGN' if 'benign' in record['image_id'] else 'MASS_MALIGNANT'
                #print("Changed MASS UNPROVEN to ", annotations.at[k, 'label'], " ", record['image_id'])
            if 'CALCIFICATION_UNPROVEN' in record['label']:
                annotations.at[k, 'label'] = 'CALCIFICATION_BENIGN' if 'benign' in record['image_id'] else 'CALCIFICATION_MALIGNANT'
                #print("Changed CALCIFICATION UNPROVEN to ", annotations.at[k, 'label'], " ", record['image_id'])
        
        
        
        annotations = annotations[annotations['label'].isin(['NORMAL', 'MASS_MALIGNANT', 'MASS_BENIGN', 'CALCIFICATION_MALIGNANT', 'CALCIFICATION_BENIGN'])]
        
        print("Number of annotations after filtering OTHER_BENIGN (18 in total) ", len(annotations))
        
        return annotations

    def __len__(self):
        # las que estan por encima de len.ddsm_annotations son patches normales 
        # en esas imagenes
        return len(self.ddsm_annotations) + self.number_of_abnormal_annotations
    
    
    def get_all_targets(self):
        """ Returns all the targets of the dataset, useful for sampling 

        Returns:
            np.array: with the targets of the dataset (per image)
        """
        labels =  self.ddsm_annotations['label'].values
        additional_labels = np.array(['NORMAL' for _ in range(self.number_of_abnormal_annotations)])
        return np.concatenate([labels, additional_labels])
        

    def __getitem__(self, idx):
        
        if idx >= len(self.ddsm_annotations):
            idx_image = idx % len(self.ddsm_annotations)
            abnormality = 'NORMAL'
        else:
            idx_image = idx
            abnormality = self.ddsm_annotations.iloc[idx]['label']
        
        image_id = self.ddsm_annotations.iloc[idx_image]['image_id']
        image_path = self.root_dir / image_id
        image = np.array(Image.open(image_path))
        
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        
        if self.normalize_input:
            image_std = image.std()
            if image_std == 0:
                print(f"Image std is 0 for image {image_path}, skipping normalization")
                image_std = 1.0
            
            image = (image - image.mean()) / image_std
            
              
        
        if self.return_mask:
            mask_id = self.ddsm_annotations.iloc[idx_image]['mask_id']
            if mask_id is None:
                mask = np.zeros_like(image, dtype=np.uint8)
            else:
                mask_path = self.root_dir / mask_id
                mask = np.array(Image.open(mask_path))
        else:
            mask = None
        
        if self.patch_sampler is None:
            if self.convert_to_rgb:
                image = np.stack((image,)*3, axis=-1)
            
            return image, abnormality, mask
        
        if self.ddsm_annotations.iloc[idx_image]['outline'] is not None:
            points = np.array(self.ddsm_annotations.iloc[idx_image]['outline']).T 
        else:
            points = None # normal image with no outline
    
        
        # image_patches, labels, mask_patches = self.patch_sampler.sample_patches(image, 
        #                                                                         abnormality, 
        #                                                                         mask= mask, 
        #                                                                         points=points)
        
        if abnormality != 'NORMAL':
            image, mask = self.patch_sampler.sample_abnormal_patch(image, mask, points)
        elif idx > idx_image:
            image, mask = self.patch_sampler.sample_hard_negative_patch(image, mask, points)
        else:
            image, mask = self.patch_sampler.sample_blob_patch(image, mask, points)
           
            
        label_idx = self.class_to_idx[abnormality] 
        
        
        # random contrast
        contrast_factor = np.random.uniform(0.8, 1.2, 1)
        image = image * contrast_factor
            
        # random shift intensity
        intensity_shift = np.random.uniform(-0.1, 0.1, 1)
        image = image + intensity_shift 
        
        image = np.expand_dims(image, axis=0)
        if self.convert_to_rgb:
            image = np.repeat(image, 3, axis=0)
        
        image = image.astype(np.float32) 
  
        if self.return_mask:
            return image, label_idx, mask
        return image, label_idx
        
        

class ConvertToFloat32:
    def __call__(self, img):
        # Convert to float and scale from 0-255 to 0-1
        img = torch.from_numpy(np.array(img)).float() 
        img.unsqueeze_(0)
        return img


class DDSM_patch_eval(datasets.DatasetFolder):
    def __init__(self, root, transform=None, 
                 convert_to_rgb = False, 
                 format_img = 'png',
                 return_mask = False, 
                 subset_size = None):
        # Call the parent constructor
        #create lambda transform function to cast to float32
        
        
        if transform is None:
            transform = transforms.Compose([ConvertToFloat32()])
        self.convert_to_rgb = convert_to_rgb
        self.return_mask = return_mask
        self.format_img = format_img
            
        def custom_loader(path):
            if '.png' in path:
                # Open the image as grayscale ('L' mode)
                img =  Image.open(path)
                return img
            elif '.npy' in path:
                return np.load(path)
            else:
                raise ValueError(f"Unknown image format {self.format_img}")
        
        extensions = ["."+format_img] 
        super().__init__(
            root,
            loader=custom_loader,
            extensions=extensions,
            transform=transform,
        )
        self.imgs = self.samples
            
            
        
        
        self.select_images() # discard the mask images
        
        # If a remap dictionary is provided, remap the class_to_idx
        self.remap_targets()
        
        if subset_size is not None:
            self.samples = self.samples[:subset_size]
            self.targets = self.targets[:subset_size]

    
    def select_images(self):
        image_suffix = '_img.png' if self.format_img == 'png' else '_img.npy'
        
        samples = [sample for sample in self.samples if image_suffix in sample[0]]
        print("Number of samples after filtering: ", len(samples))
        self.samples = samples
        self.targets = [sample[1] for sample in samples]
        
    
    def remap_targets(self):
        """
        Remap the class_to_idx dictionary based on a user-defined mapping.
        """
        mapping_names = {'background': 0,
            'benign_mass': 1,
            'benign_calc': 2,
            'malignant_mass': 3,
            'malignant_calc': 4,
        }
        
        class_names = ['NORMAL',
            'MASS_BENIGN',
            'CALCIFICATION_BENIGN',
            'MASS_MALIGNANT',
            'CALCIFICATION_MALIGNANT',
        ]
        self.class_to_idx = {class_name: i for i, class_name in enumerate(class_names)}
        self.idx_to_class = {i: class_name for i, class_name in enumerate(class_names)}
        
        # translate all the labels

        for k, (path, label) in enumerate(self.samples):
            sample_class = mapping_names[path.split('/')[-2]]
            
            self.samples[k] = (path, sample_class)    
            self.targets[k] = sample_class
            
    def __len__(self):
        return len(self.samples)     
            
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        
        if self.transform is not None:
            sample = self.transform(sample)
            
        if self.convert_to_rgb:
            sample = sample.repeat(3, 1, 1)
        
        if self.return_mask:
            mask_path = path.replace(f'_img.{self.format_img}', '_mask.png')
            mask = self.loader(mask_path)
            mask = np.array(mask)

            return sample, target, mask
        else:
            return sample, target
        
        
def collate_fn(batch):
    # Unzip the batch into two lists: patch tensors and label tensors
    if len(batch[0]) == 2:
        patch_images, patch_labels = zip(*batch)
        mask_patches = None
    else:
        patch_images, patch_labels, mask_patches = zip(*batch)
    
    # concatentate all patch tensors along the first dimension (n_all_patches)
    patch_images = np.concatenate(patch_images, axis=0)
    
    
    if mask_patches is not None:
        mask_patches = np.concatenate(mask_patches, axis=0)
        mask_patches = torch.tensor(mask_patches, dtype=torch.uint8)
        
    patch_labels = np.concatenate(patch_labels, axis=0) 
    
    if patch_images.dtype == np.uint8:
        patch_images = torch.tensor(patch_images, dtype=torch.float32) / 255.0
    elif patch_images.dtype == np.int16:
        patch_images = torch.tensor(patch_images, dtype=torch.float32) / 32767.0
    else:
        patch_images = torch.tensor(patch_images, dtype=torch.float32)
        
    patch_labels = torch.tensor(patch_labels, dtype=torch.long)
    
    if mask_patches is not None:
        return patch_images, patch_labels, mask_patches
    return patch_images, patch_labels


def get_train_dataloader(split_csv, ddsm_annotations, root_dir, patch_size, batch_size=32, 
                         convert_to_rgb = True, shuffle=True, num_workers=4, return_mask=False, 
                         subset_size=None, include_normals=True, normalize_input = False): 
    
    
    affine_transform = RandomAffineTransform()
    #affine_transform = IdentityTransform()


    patch_sampler = PatchSampler(patch_size, affine_transform=affine_transform,
                                n_positive_crops = 1, 
                                n_hard_negative_crop=1,
                                n_blob_crops=0,
                                n_random_crops=0)

    dataset = DDSM_Patch_Dataset(split_csv, ddsm_annotations, root_dir, 
                                return_mask= return_mask, patch_sampler = patch_sampler,
                                convert_to_rgb = convert_to_rgb,
                                subset_size = subset_size,
                                include_normals=include_normals,
                                normalize_input = normalize_input)

    dataloader_sampler = BalancedPatchBatchSampler(dataset, batch_size)

    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=dataloader_sampler)
    return dataloader

def get_test_dataloader(patches_root, batch_size=32, return_mask=False, convert_to_rgb = True, subset_size=None, format_img = 'png'):
    dataset = DDSM_patch_eval(patches_root, return_mask=return_mask, convert_to_rgb= convert_to_rgb, subset_size=subset_size, format_img=format_img)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return dataloader



# Data Module to handle data loading and transformations
class DDSMPatchDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.source_root = get_parameter(config, ['General', 'source_root'], default=None)
        
        self.batch_size = get_parameter(config, ['Datamodule',  'batch_size'])
        self.num_workers = get_parameter(config, ['Datamodule', 'num_workers'])
        
        self.ddsm_root = get_parameter(config, ['Datamodule', 'train_set','ddsm_root'])
        self.split_csv = get_parameter(config, ['Datamodule', 'train_set', 'split_csv'])
        self.ddsm_annotations = get_parameter(config, ['Datamodule', 'train_set', 'ddsm_annotations'])
        self.patch_size = get_parameter(config, ['Datamodule', 'train_set', 'patch_size'])
        self.convert_to_rgb = get_parameter(config, ["Datamodule", 'train_set', "convert_to_rgb"], default=True)
        self.normalize_input = get_parameter(config, ["Datamodule",'train_set', "normalize_input"], default=False)   
        self.subset_size_train = get_parameter(config, ['Datamodule', 'train_set','subset_size_train'], default=None)
        self.include_normals = get_parameter(config, ['Datamodule','train_set', 'include_normals'], default=True)
        
        
        self.eval_patches_root = get_parameter(config, ['Datamodule', 'val_set', 'eval_patches_root'])
        self.subset_size_test = get_parameter(config, ['Datamodule', 'val_set', 'subset_size_test'], default=None)
        
        self.return_mask = True
        
        
        self.source_root = pathlib.Path(self.source_root) if self.source_root is not None else None
        print("Source root: ", self.source_root)
        print("split csv: ", self.split_csv)
        print("ddsm annotations: ", self.ddsm_annotations)
        self.split_csv = self.source_root / self.split_csv if self.source_root is not None else self.split_csv
        self.ddsm_annotations = self.source_root / self.ddsm_annotations if self.source_root is not None else self.ddsm_annotations
        
        assert str(self.patch_size) in str(self.eval_patches_root), "eval_patches should be of the same size as the training patches: " + str(self.patch_size)
        
        

    def prepare_data(self):
        # Download or prepare the data if needed
        assert pathlib.Path(self.ddsm_root).exists(), f"Data directory {self.ddsm_root} does not exist."
        assert pathlib.Path(self.split_csv).exists(), f"Split CSV file {self.split_csv} does not exist."
        assert pathlib.Path(self.ddsm_annotations).exists(), f"Annotations file {self.ddsm_annotations} does not exist."
        
    
    def setup(self, stage=None):
        # Create train, val, test datasets
        pass # everythin is in get_train_dataloader and get_test_dataloader
    
    def train_dataloader(self):
        return get_train_dataloader(self.split_csv, 
                                    self.ddsm_annotations, 
                                    self.ddsm_root, 
                                    patch_size=self.patch_size,
                                    batch_size=self.batch_size, 
                                    convert_to_rgb=self.convert_to_rgb,
                                    shuffle=True, num_workers=self.num_workers, 
                                    return_mask=self.return_mask, subset_size=self.subset_size_train,
                                    include_normals=self.include_normals,
                                    normalize_input = self.normalize_input)
    
    def val_dataloader(self):
        return get_test_dataloader(self.eval_patches_root, batch_size=self.batch_size, 
                                   convert_to_rgb=self.convert_to_rgb,
                                   return_mask=self.return_mask, subset_size=self.subset_size_test,
                                   format_img = 'npy')
        
    
    def test_dataloader(self):
        return get_test_dataloader(self.eval_patches_root, batch_size=self.batch_size, return_mask=False)
    
    
# Dataset to load full ddsm images
class DDSM_Image_Dataset(Dataset):
    def __init__(self, split_csv, ddsm_annotations, root_dir, 
                convert_to_rgb = True,
                return_mask=False,
                subset_size = None, 
                random_seed = 42,
                num_normal_images_test = 700,
                geometrical_transform = None,
                intensity_transform = None,
                use_all_images = False):
            
        
        self.split_csv = split_csv
        self.root_dir = pathlib.Path(root_dir)
        self.return_mask = return_mask
        self.convert_to_rgb = convert_to_rgb
        self.random_seed = random_seed
        self.num_normal_images_test = num_normal_images_test
        self.geometrical_transform = geometrical_transform
        self.intensity_transform = intensity_transform
        self.use_all_images = use_all_images # use all images in the dataset including not annotated in cancer/benign folders
        
        self.ddsm_annotations = self.load_annotations(split_csv, ddsm_annotations)
        
        if subset_size is not None:
            print("Subsetting dataset to ", subset_size)
            self.ddsm_annotations = self.ddsm_annotations.sample(subset_size)
        
        #class_names = self.ddsm_annotations['label'].unique()
        class_names = ['NORMAL',
            'CANCER',
        ]
        self.class_to_idx = {class_name: i for i, class_name in enumerate(class_names)}
        self.idx_to_class = {i: class_name for i, class_name in enumerate(class_names)}
        
    
    def get_all_targets(self):
        """ Returns all the targets of the dataset, useful for sampling 

        Returns:
            np.array: with the targets of the dataset (per image)
        """
        return self.ddsm_annotations['breast_malignant'].values
    
    def sample_equiprobable(self):
        positive = np.random.rand() > 0.5
        
        labels = self.ddsm_annotations['breast_malignant'].values
            
        if positive:
            positive_idx = np.random.choice(np.where(labels == True)[0])
            return positive_idx
        else:
            negative_idx = np.random.choice(np.where(labels == False)[0])
            return negative_idx
        

    def load_annotations(self, split_csv,  annotations_file):
        
        split_images = pd.read_csv(split_csv)
        print("Number of images in split: ", len(split_images))
       
        if str(annotations_file).endswith('.json'):
            annotations = pd.read_json(annotations_file, orient='records', lines=True)
        else:
            with gzip.open(annotations_file, 'rt', encoding='utf-8') as f:
                annotations = pd.read_json(f, orient='records', lines=True)

       
        annotations_final = []
        imagenes_sin_anot = 0
        
        for image_id in split_images['ddsm_image']:
            anot = {}
            if image_id in annotations.image_id.values:
                anot['image_id'] = image_id
                anot['mask_id'] = image_id.replace('.png', '_totalmask.png')
                anot['breast_malignant']= annotations[annotations['image_id'] == image_id]['breast_malignant'].values[0]
                annotations_final.append(anot)
            elif 'normal' in image_id:
                anot['image_id'] = image_id
                anot['mask_id'] = None
                anot['breast_malignant'] = False
                annotations_final.append(anot)
            else:
                #print(f"Image {image_id} not found in annotations and not notmals")
                #imagenes en directorios posiblemente con cancer pero que no tienen anotaciones
                # las anomalias estn en el otro pecho
                # las quitamos de momento porque tenemos muchas mas normales
                # y las benignas tambien no son cancer
                imagenes_sin_anot += 1
                if self.use_all_images:
                    anot['image_id'] = image_id
                    anot['mask_id'] = None
                    anot['breast_malignant'] = False
                    annotations_final.append(anot)
                
            
            
        print("Number of images after assgining labels: ", len(annotations_final), " Images without annotations (other breasts in cancer/bening folders): ", imagenes_sin_anot)
        
        return pd.DataFrame(annotations_final)

        # print("Number of annotations: ", len(annotations))
        
        
        # # filter all annotations that are in the train_images
        # annotations = annotations[annotations['image_id'].isin(split_images['ddsm_image'])]
        # print("Number of annotations after filtering split: ", len(annotations))
        
        
        #     # annotations only contains abnormal images, lets add normal images for training
        # normals_images = [im for im in split_images['ddsm_image'].values if 'normals' in im]
        
        # train = True if 'train' in str(split_csv).lower() else False
        
        # # if train:
        # #     normal_images = normals_images[:-self.num_normal_images_test]
        # # else:
        # #     print("num normal images test: ", len(normals_images))
        # #     normal_images = normals_images[-self.num_normal_images_test:]
        
        # print(f"Including {len(normals_images)} normal images")
        
        # normal_records = []
        # for normal_image in normals_images:
        #     record = {
        #         'type': "NORMAL",
        #         'assessment': None,
        #         'subtlety': None,
        #         'pathology': "NORMAL",
        #         'outline' : None,
        #         'bounding_box': None,
        #         'breast_malignant': False,
        #         'image_id': normal_image,
        #         'mask_id': None
        #     }
        #     normal_records.append(record)
        
        # annotations = pd.concat([annotations, pd.DataFrame(normal_records)], ignore_index=True)
        
        # print("Number of annotations after adding normals: ", len(annotations))
 
        # annotations = annotations.groupby('image_id').first().reset_index()
        
        # print("Number of annotations after removing repeated: ", len(annotations))
        
        # return annotations
        
    def __len__(self):
        return len(self.ddsm_annotations)
    
    def __getitem__(self, idx):
        image_id = self.ddsm_annotations.iloc[idx]['image_id']
        image_path = self.root_dir / image_id
        image = np.array(Image.open(image_path)).astype(np.float32)
        
        
        if self.return_mask:
            mask_id = self.ddsm_annotations.iloc[idx]['mask_id']
            if mask_id is None:
                mask = np.zeros_like(image, dtype=np.uint8)
            else:
                mask_path = self.root_dir / mask_id
                if mask_path.exists():
                    
                    mask = np.array(Image.open(mask_path))
                    
                else:
                    mask = np.zeros_like(image, dtype=np.uint8)
        else:
            mask = None
            
 
        
        label = int(self.ddsm_annotations.iloc[idx]['breast_malignant'])
        
        if self.geometrical_transform is not None:
            if mask is not None:
                augmented = self.geometrical_transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
            else:
                image = self.geometrical_transform(image)
                
        
        if self.convert_to_rgb:
            image = np.stack((image,)*3, axis=-1)
        else:
            image = np.expand_dims(image, axis=0)
        
            
        image = torch.from_numpy(image).float()   
        
        if self.intensity_transform is not None:
            image = self.intensity_transform(image)
      
                
        if mask is not None:
            
            p = np.random.rand()
            if p < 0.1 and label == 1:
                background = (mask == 0).astype(float)
                background = np.expand_dims(background, axis=0)
                background = torch.from_numpy(background).float()
                image = image * background
                label = 0         
            return image, label, mask, pathlib.Path(image_id).stem
        return image, label, pathlib.Path(image_id).stem
    


class DDSM_Image_Dataset_mixup(DDSM_Image_Dataset):
    def __init__(self, split_csv, ddsm_annotations, root_dir, 
            convert_to_rgb = True,
            return_mask=False,
            subset_size = None, 
            random_seed = 42,
            num_normal_images_test = 700,
            geometrical_transform = None,
            intensity_transform = None,
            mixup_alpha = 0.4,
            use_all_images = False):
        

        super().__init__(split_csv, ddsm_annotations, root_dir,
            convert_to_rgb = convert_to_rgb,
            return_mask=return_mask,
            subset_size = subset_size,
            random_seed = random_seed,
            num_normal_images_test = num_normal_images_test,
            geometrical_transform = geometrical_transform,
            intensity_transform = intensity_transform,
            use_all_images = use_all_images)
        
        self.mixup_alpha = mixup_alpha
        
    
    def __getitem__(self, idx):
        image, label1, mask, image_id = super().__getitem__(idx)
                
        # mixup
        idx2 = self.sample_equiprobable()
        image2, label2, mask2, image_id2 = super().__getitem__(idx2)
        
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        image = lam * image + (1 - lam) * image2
        if mask is not None:
            mask = lam * mask + (1 - lam) * mask2
            
        label = (label1, label2, lam)
            
        return image, label, mask, image_id
    

class DreamPilot_Image_Dataset(Dataset):
    def __init__(self, dream_pilot_folder, transform = None):
        self.dream_pilot_folder = pathlib.Path(dream_pilot_folder)
        self.transform = transform
        
        self.image_crosswalk = pd.read_csv(self.dream_pilot_folder / 'images_crosswalk.tsv', sep='\t')
        
    def __len__(self):
        return len(self.image_crosswalk)
    
    def __getitem__(self, idx):
        image_id = self.image_crosswalk.iloc[idx]['filename'].replace('.dcm', '.png')
        image_path = self.dream_pilot_folder / image_id
        image = Image.open(image_path)
        
        label = self.image_crosswalk.iloc[idx]['cancer']
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label
        

# Data Module to handle data loading and transformations

class DDSMImageDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.source_root = get_parameter(config, ['General', 'source_root'], default=None)
        self.source_root = pathlib.Path(self.source_root) if self.source_root is not None else None

        self.ddsm_root = get_parameter(config, ['Datamodule', 'ddsm_root'])
        self.train_csv = get_parameter(config, ['Datamodule', 'train_csv'])
        self.val_csv = get_parameter(config, ['Datamodule', 'val_csv'])
        self.ddsm_annotations = get_parameter(config, ['Datamodule', 'ddsm_annotations'])
        self.ddsm_annotations = self.source_root / self.ddsm_annotations if self.source_root is not None else self.ddsm_annotations
        self.convert_to_rgb = get_parameter(config, ["Datamodule", "convert_to_rgb"], default=True)
        self.return_mask = get_parameter(config, ["Datamodule", "return_mask"], default=True)
        self.subset_size_train = get_parameter(config, ['Datamodule', 'subset_size_train'], default=None)
        self.subset_size_test = get_parameter(config, ['Datamodule', 'subset_size_test'], default=None)
        self.batch_size = get_parameter(config, ['Datamodule', 'batch_size'])
        self.balanced_patches = get_parameter(config, ['Datamodule', 'balanced_patches'], default=False)
        self.num_workers = get_parameter(config, ['Datamodule', 'num_workers'])
        self.mixup_alpha = get_parameter(config, ['Datamodule', 'mixup_alpha'], default=None) # use mixup dataset for training
        
        self.train_csv = self.source_root / self.train_csv if self.source_root is not None else self.train_csv
        self.val_csv = self.source_root / self.val_csv if self.source_root is not None else self.val_csv
        
        
        #self.num_normal_images_test = get_parameter(config, ['Datamodule', 'num_normal_images_test'], default=700)
        self.random_seed = get_parameter(config, ['Datamodule', 'random_seed'], default=42)
        self.dream_pilot_folder = get_parameter(config, ['Datamodule', 'dream_pilot_folder'], default=None)
        
        
    def prepare_data(self):
        # Download or prepare the data if needed
        assert pathlib.Path(self.ddsm_root).exists(), f"Data directory {self.ddsm_root} does not exist."
        assert pathlib.Path(self.train_csv).exists(), f"Split CSV file {self.train_csv} does not exist."
        assert pathlib.Path(self.val_csv).exists(), f"Split CSV file {self.val_csv} does not exist."
        assert pathlib.Path(self.ddsm_annotations).exists(), f"Annotations file {self.ddsm_annotations} does not exist."
        if self.dream_pilot_folder is not None:
            assert pathlib.Path(self.dream_pilot_folder).exists(), f"Dream pilot folder {self.dream_pilot_folder} does not exist."
        
    
    def setup(self, stage=None):
        # Create train, val, test datasets
        pass # everythin is in get_train_dataloader and get_test_dataloader
    
    def train_dataloader(self):
        
        # geometrical_transform = transforms.Compose([
        #     transforms.RandomAffine(degrees=15, shear=10, scale=(0.8, 1.2)),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
            
        # ])
        
        

        # Define transformations
        geometrical_transform = A.Compose([
            A.Affine(scale=(0.8, 1.2), shear=10, rotate=15, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
        ], additional_targets={'mask': 'mask'})
        
        
        
        if self.convert_to_rgb:
            intensity_transform = transforms.Compose([
                Standardize(),
                RandomIntensity(0.8, 1.2),
                RandomContrast(0.8, 1.2),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1))])
        else:
            intensity_transform = Standardize()

    

        if self.mixup_alpha is None:
            dataset = DDSM_Image_Dataset(self.train_csv, self.ddsm_annotations, self.ddsm_root, 
                                        convert_to_rgb=False, 
                                        subset_size=self.subset_size_train, random_seed=self.random_seed,
                                        geometrical_transform=geometrical_transform,
                                        intensity_transform=intensity_transform,
                                        return_mask=self.return_mask,
                                        use_all_images=True)  
        else:
            raise NotImplementedError("Mixup not implemented")
            # dataset = DDSM_Image_Dataset_mixup(self.train_csv, self.ddsm_annotations, self.ddsm_root, 
            #                             convert_to_rgb=False, 
            #                             subset_size=self.subset_size_train, random_seed=self.random_seed,
            #                             geometrical_transform=geometrical_transform,
            #                             intensity_transform=intensity_transform,
            #                             return_mask=self.return_mask,
            #                             mixup_alpha=self.mixup_alpha,
            #                             use_all_images=True)
        
        if self.balanced_patches:
            print("Using balanced batch sampler")
            sampler = BalancedBatchSampler(dataset, batch_size=self.batch_size)
        else:
            sampler = None
           
        dataloader = DataLoader(dataset, batch_size=self.batch_size, 
                                sampler=sampler, num_workers=self.num_workers)
        return dataloader
    
    def val_dataloader(self):
        geometrical_transform = None
        
        if self.convert_to_rgb:
            intensity_transform = transforms.Compose([
                Standardize(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1))])
        else:
            intensity_transform = Standardize()

        dataset = DDSM_Image_Dataset(self.val_csv, self.ddsm_annotations, self.ddsm_root, 
                                    convert_to_rgb=False, 
                                    subset_size=self.subset_size_test, random_seed=self.random_seed,
                                    geometrical_transform=geometrical_transform,
                                    intensity_transform=intensity_transform,
                                    return_mask=self.return_mask,
                                    use_all_images=True)  
        
        test_batch_size = self.batch_size  # so we can TTA easily
        dataloader = DataLoader(dataset, batch_size=test_batch_size, shuffle=False, num_workers=self.num_workers)
        return dataloader
    
    def test_dataloader(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            Standardize()
        ])
        
        if self.convert_to_rgb:
            transform.transforms.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
            
        
        dataset = DreamPilot_Image_Dataset(self.dream_pilot_folder, transform=transform)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return dataloader
