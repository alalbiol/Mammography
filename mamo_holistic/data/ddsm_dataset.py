import os
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



from utils.sample_patches_main import sample_positive_bb, sample_negative_bb,  sample_hard_negative_bb, sample_blob_negative_bb

class RandomAffineTransform:
    def __init__(self, angle_range=(-180, 180), shear_range=(-0., 0.), 
                 scale_range=(0.9, 1.1)):
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
                 return_mask=False,
                 patch_sampler = None):
        
        self.split_csv = split_csv
        self.root_dir = pathlib.Path(root_dir)
        self.return_mask = return_mask
        self.patch_sampler = patch_sampler
        
        self.ddsm_annotations = self.load_annotations(split_csv, root_dir, ddsm_annotations)
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
        

    def load_annotations(self, split_csv, root_dir,  annotations_file):
        
        split_images = pd.read_csv(split_csv)
        
        annotations = pd.read_json('../resources/ddsm/ddsm_annotations.json', orient='records', lines=True)
        print("Number of annotations: ", len(annotations))
        
        
        # filter all annotations that are in the train_images
        annotations = annotations[annotations['image_id'].isin(split_images['ddsm_image'])]
        print("Number of annotations after filtering split: ", len(annotations))
        
        
        # annotations only contains abnormal images, lets add normal images for training
        normals_images = [im for im in split_images['ddsm_image'].values if 'normals' in im]
        print(f"Found {len(normals_images)} normal images")
        
        normal_records = []
        for normal_image in normals_images:
            record = {
                'type': "NORMAL",
                'assesment': None,
                'subtlety': None,
                'pathology': "NORMAL",
                'outline' : None,
                'bounding_box': None,
                'breast_malignat': False,
                'image_id': normal_image,
                'mask_id': None
            }
            normal_records.append(record)
            
        annotations = pd.concat([annotations, pd.DataFrame(normal_records)], ignore_index=True)
            
        annotations['label'] = annotations['type'] + "_" + annotations['pathology']
        
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
        
        
        print("Number of annotations after adding normals: ", len(annotations))
        
        annotations = annotations[annotations['label'].isin(['NORMAL', 'MASS_MALIGNANT', 'MASS_BENIGN', 'CALCIFICATION_MALIGNANT', 'CALCIFICATION_BENIGN'])]
        
        print("Number of annotations after filtering OTHER_BENIGN (18 in total) ", len(annotations))
        
        return annotations


    def __len__(self):
        return len(self.ddsm_annotations)

    def __getitem__(self, idx):
        
        abnormality = self.ddsm_annotations.iloc[idx]['label']
        
        image_id = self.ddsm_annotations.iloc[idx]['image_id']
        image_path = self.root_dir / image_id
        image = np.array(Image.open(image_path))
        
        if self.return_mask:
            mask_id = self.ddsm_annotations.iloc[idx]['mask_id']
            if mask_id is None:
                mask = np.zeros_like(image, dtype=np.uint8)
            else:
                mask_path = self.root_dir / mask_id
                mask = np.array(Image.open(mask_path))
        else:
            mask = None
        
        if self.patch_sampler is None:
            return image, abnormality, mask
        
        if self.ddsm_annotations.iloc[idx]['outline'] is not None:
            points = np.array(self.ddsm_annotations.iloc[idx]['outline']).T 
        else:
            points = None # normal image with no outline
    
        
        image_patches, labels, mask_patches = self.patch_sampler.sample_patches(image, abnormality, mask= mask, points=points)
        
        labels_idx = [self.class_to_idx[label] for label in labels]
        
        image_patches = np.array(image_patches)
        
        if mask is not None:
            mask_patches = np.array(mask_patches)
        else:
            mask_patches = None
        labels_idx = np.array(labels_idx)
        
        return image_patches, labels_idx, mask_patches
        
        

class ConvertToFloat32:
    def __call__(self, img):
        # Convert to float and scale from 0-255 to 0-1
        img = torch.from_numpy(np.array(img)).float() / 255.0
        img.unsqueeze_(0)
        return img


class DDSM_patch_eval(datasets.ImageFolder):
    def __init__(self, root, transform=None, return_mask = False):
        # Call the parent constructor
        #create lambda transform function to cast to float32
        
        
        if transform is None:
            transform = transforms.Compose([ConvertToFloat32()])
        self.return_mask = return_mask
            
        def custom_loader(path):
            # Open the image as grayscale ('L' mode)
            return Image.open(path)
    
            
        super().__init__(root, transform=transform, loader=custom_loader)
        
        self.select_images() # discard the mask images
        
        # If a remap dictionary is provided, remap the class_to_idx
        self.remap_targets()

    
    def select_images(self):
        samples = [sample for sample in self.samples if '_img.png' in sample[0]]
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
            
            
            
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        
        if self.transform is not None:
            sample = self.transform(sample)
        
        if self.return_mask:
            mask_path = path.replace('_img.png', '_mask.png')
            mask = self.loader(mask_path)
            if self.transform is not None:
                mask = self.transform(mask)
            return sample, target, mask
        else:
            return sample, target, None
        
        
def collate_fn(batch):
    # Unzip the batch into two lists: patch tensors and label tensors
    patch_images, patch_labels, mask_patches = zip(*batch)
    
    # concatentate all patch tensors along the first dimension (n_all_patches)
    patch_images = np.concatenate(patch_images, axis=0)
    
    
    if mask_patches[0] is not None:
        mask_patches = np.concatenate(mask_patches, axis=0)
        
    patch_labels = np.concatenate(patch_labels, axis=0) 
    
    if patch_images.dtype == np.uint8:
        patch_images = torch.tensor(patch_images, dtype=torch.float32) / 255.0
    else:
        patch_images = torch.tensor(patch_images, dtype=torch.float32)
    # add channel dimension
    patch_images = patch_images.unsqueeze(1)
    
    if mask_patches[0] is not None:
        mask_patches = torch.tensor(mask_patches, dtype=torch.uint8)
    else:
        mask_patches = None
        
    patch_labels = torch.tensor(patch_labels, dtype=torch.long)
    
    return patch_images, patch_labels, mask_patches



def get_train_dataloader(split_csv, ddsm_annotations, root_dir, batch_size=32, shuffle=True, num_workers=4, return_mask=False):
    
    
    affine_transform = RandomAffineTransform()
    #affine_transform = IdentityTransform()


    patch_sampler = PatchSampler(512, affine_transform=affine_transform,
                                n_positive_crops = 1, 
                                n_hard_negative_crop=1,
                                n_blob_crops=1,
                                n_random_crops=1)

    dataset = DDSM_Patch_Dataset(split_csv, ddsm_annotations, root_dir, 
                                return_mask= return_mask, patch_sampler = patch_sampler)

    

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    return dataloader

def get_test_dataloader(patches_root, batch_size=32, return_mask=False):
    dataset = DDSM_patch_eval(patches_root, return_mask=return_mask)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return dataloader



# Data Module to handle data loading and transformations
class DDSMPatchDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.split_csv = get_parameter(config, ['Datamodule', 'split_csv'])
        self.ddsm_annotations = get_parameter(config, ['Datamodule', 'ddsm_annotations'])
        self.ddsm_root = get_parameter(config, ['Datamodule', 'ddsm_root'])
        self.batch_size = get_parameter(config, ['Datamodule', 'batch_size'])
        self.num_workers = get_parameter(config, ['Datamodule', 'num_workers'])
        self.eval_patches_root = get_parameter(config, ['Datamodule', 'eval_patches_root'])
        
        
        

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
                                    batch_size=self.batch_size, 
                                    shuffle=True, num_workers=self.num_workers, return_mask=False)
    
    def val_dataloader(self):
        return get_test_dataloader(self.eval_patches_root, batch_size=self.batch_size, return_mask=False)
        
    
    def test_dataloader(self):
        return get_test_dataloader(self.eval_patches_root, batch_size=self.batch_size, return_mask=False)