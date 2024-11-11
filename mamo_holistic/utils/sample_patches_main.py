import numpy as np
import cv2
import pandas as pd
import os, sys, argparse
from PIL import Image
import pathlib
from shapely.geometry import Polygon, Point



def overlap_bb_boundary(bb, boundary_points, cutoff=.5):
    # List of points defining the object boundary
    object_polygon = Polygon(boundary_points)
    
    if object_polygon.is_valid == False:
        # print("Invalid polygon")
        # np.savetxt("invalid_polygon.csv", boundary_points, delimiter=",")       
        object_polygon = object_polygon.buffer(0)

    # Bounding box coordinates
    bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max = bb[0], bb[1], bb[2], bb[3]
    bbox_polygon = Polygon([
        (bbox_x_min, bbox_y_min),
        (bbox_x_max, bbox_y_min),
        (bbox_x_max, bbox_y_max),
        (bbox_x_min, bbox_y_max)
    ])

    # Calculate intersection and union
    inter_area = object_polygon.intersection(bbox_polygon).area
    patch_area = object_polygon.area
    bb_area = bbox_polygon.area
    
    if patch_area == 0:
        return False
    if bb_area == 0:
        return False
    
    return (inter_area/bb_area > cutoff or inter_area/patch_area > cutoff)
    

def point_in_polygon(point, polygon):
    """Check if a point is inside a polygon.
    Args:
        point (list): [x, y]
        polygon (list): [(x1, y1), (x2, y2), ...]
    Returns:
        bool: True if point is inside polygon
    """
    x, y = point
    return polygon.contains(Point(x, y))

def overlap_patch_roi(patch_center, patch_size, roi_mask, 
            add_val=1000, cutoff=.5):
    x1,y1 = (patch_center[0] - patch_size//2, 
            patch_center[1] - patch_size//2)
    x2,y2 = (patch_center[0] + patch_size//2, 
            patch_center[1] + patch_size//2)
    x1 = np.clip(x1, 0, roi_mask.shape[1])
    y1 = np.clip(y1, 0, roi_mask.shape[0])
    x2 = np.clip(x2, 0, roi_mask.shape[1])
    y2 = np.clip(y2, 0, roi_mask.shape[0])
    roi_area = (roi_mask>0).sum() 
    roi_patch_added = roi_mask.copy()
    
    
    #print(f"{x1=} {x2=} {y1=} {y2=}")
    roi_patch_added[y1:y2, x1:x2] += add_val
    patch_area = (roi_patch_added>=add_val).sum()
    inter_area = (roi_patch_added>add_val).sum().astype('float32')
    
    if patch_area == 0:
        return False
    if roi_area == 0:
        return False
    
    return (inter_area/roi_area > cutoff or inter_area/patch_area > cutoff)

def create_blob_detector(roi_size=(128, 128), blob_min_area=3, 
                        blob_min_int=.5, blob_max_int=.95, blob_th_step=10):
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = blob_min_area
    params.maxArea = roi_size[0]*roi_size[1]
    params.filterByCircularity = False
    params.filterByColor = False
    params.filterByConvexity = False
    params.filterByInertia = False
    # blob detection only works with "uint8" images.
    params.minThreshold = int(blob_min_int*255)
    params.maxThreshold = int(blob_max_int*255)
    params.thresholdStep = blob_th_step
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3:
        return cv2.SimpleBlobDetector(params)
    else:
        return cv2.SimpleBlobDetector_create(params)


def add_img_margins(img, margin_size):
    '''Add all zero margins to an image
    '''
    enlarged_img = np.zeros((img.shape[0]+margin_size*2, 
                            img.shape[1]+margin_size*2))
    enlarged_img[margin_size:margin_size+img.shape[0], 
                margin_size:margin_size+img.shape[1]] = img
    return enlarged_img

def sample_patches(img, roi_mask, out_dir, mask_id, bounding_box, patch_size=256,
                pos_cutoff=.75, neg_cutoff=.35,
                nb_bkg=100, nb_abn=100, 
                bkg_dir='background', 
                malignant_mass_dir='malignant_mass', 
                malignant_calc_dir='malignant_calc',
                bening_mass_dir='benign_mass',
                bening_calc_dir='benign_calc', 
                save_patch_mask=True,
                verbose=False):

    """Sample positive and negative patches from an image. 
    
    Positive patches are randomly sampled and accepted if the overlap with the ROI is over pos_cutoff.
    Negative patches are randomly sampled and accepted if the overlap with the ROI is below neg_cutoff.
    
    Args:
    img: numpy array, the full image.
    roi_mask: numpy array, the ROI mask.
    out_dir: str, the output directory.
    mask_id: str, the mask ID. Used to name patches and extract pathology and type of lession.
    bounding_box: list, the bounding box of the ROI. (xmin, ymin, w, h)
    patch_size: int, the patch size.
    pos_cutoff: float, the overlap cutoff for positive patches.
    neg_cutoff: float, the overlap cutoff for negative patches.
    nb_bkg: int, the number of background patches.
    nb_abn: int, the number of abnormal patches.
    bkg_dir: str, the background directory.
    malignant_dir: str, the malignant directory.
    bening_dir: str, the benign directory.
    verbose: bool, print out debug info.

    Returns:
    abn_bbs: list with abnormal bounding boxes. [xmin, ymin, w, h] 
    bkg_bbs: list with background bounding boxes. [xmin, ymin, w, h]
    """
    
    out_dir = pathlib.Path(out_dir)
    
    
    pathology = "MALIGNANT" if "MALIGNANT" in mask_id else "BENIGN"
    type = "MASS" if "MASS" in mask_id else "CALCIFICATION"
    
    
    if pathology == "MALIGNANT" and type == "MASS":
        abn_dir = out_dir / malignant_mass_dir
    elif pathology == "MALIGNANT" and type == "CALCIFICATION":
        abn_dir = out_dir / malignant_calc_dir
    elif pathology == "BENIGN" and type == "MASS":
        abn_dir = out_dir / bening_mass_dir
    elif pathology == "BENIGN" and type == "CALCIFICATION":
        abn_dir = out_dir / bening_calc_dir
    else:
        raise ValueError("Pathology or type not recognized.")
    
    bkg_dir = out_dir / bkg_dir
    
    if not abn_dir.exists():
        abn_dir.mkdir(parents=True)
    if not bkg_dir.exists():
        bkg_dir.mkdir(parents=True)
    
    basename = mask_id.split('/')[-1].replace('.png', '') #D_4124_1.RIGHT_CC_MASS_MALIGNANT_mask_0
    rx,ry,rw,rh = round(bounding_box[0]),round(bounding_box[1]), round(bounding_box[2]), round(bounding_box[3])

    assert rw > 0, "rw > 0"
    assert rh > 0, "rh > 0"
    assert rx >= 0, "rx >= 0"
    assert ry >= 0, "ry >= 0"
    assert rx + rw < img.shape[1], f"rx + rw < img.shape[1], {rx + rw} < {img.shape[1]}"
    assert ry + rh < img.shape[0], f"ry + rh < img.shape[0], {ry + rh} < {img.shape[0]}"
    
    
    img = add_img_margins(img, patch_size//2)
    roi_mask = add_img_margins(roi_mask, patch_size//2)

    abn_bbs = [] # abnormal bounding boxes 
    bkg_bbs = [] # background bounding boxes

    
    
    # Sample abnormality first.
    sampled_abn = 0
    nb_try = 0
    while sampled_abn < nb_abn:
        x = np.random.randint(rx, rx + rw)
        y = np.random.randint(ry, ry + rh)
        nb_try += 1
        if nb_try >= 1000:
            print("Nb of trials reached maximum, decrease overlap cutoff by 0.05")
            sys.stdout.flush()
            pos_cutoff -= .05
            nb_try = 0
            if pos_cutoff <= .0:
                raise Exception("overlap cutoff becomes non-positive, "
                                "check roi mask input.")
        # import pdb; pdb.set_trace()
        if overlap_patch_roi((x+patch_size //2,y+patch_size //2), patch_size, roi_mask, cutoff=pos_cutoff):
            patch = img[y:y + patch_size, 
                        x:x + patch_size]
            
            assert x + patch_size < img.shape[1], "x + patch_size < img.shape[1]"
            assert y + patch_size < img.shape[0], "y + patch_size < img.shape[0]"
            assert x >= 0, "x >= 0"
            assert y >= 0, "y >= 0" 
            
            abn_bbs.append((x, y, patch_size, patch_size))
            patch_img = Image.fromarray(patch.astype('int32'), mode='I')
            filename = f"{basename}_{sampled_abn:04d}_img.png"
            fullname = abn_dir / filename
            
            # print(fullname)
            # print(patch_img.size)
            # print(x, y, img.shape)
            # print(f"{rx=} {ry=} {rw=} {rh=}")
            
            patch_img.save(str(fullname))
        
            
            if save_patch_mask:
                patch_mask = roi_mask[y:y + patch_size,
                                    x:x + patch_size]
                
                
                patch_mask_img = Image.fromarray(patch_mask.astype('uint8'))
                filename = f"{basename}_{sampled_abn:04d}_mask.png"
                fullname = abn_dir / filename
                patch_mask_img.save(str(fullname))
            
            
            sampled_abn += 1
            nb_try = 0
            if verbose:
                print(f"{filename}: sampled abn at", (x,y))
    # Sample background.
    sampled_bkg = 0
    while sampled_bkg < nb_bkg:
        x = np.random.randint(0, img.shape[1] - patch_size) # rememeber that the image has been enlarged with margins
        y = np.random.randint(0, img.shape[0] - patch_size)
        if not overlap_patch_roi((x+patch_size //2,y+patch_size //2), patch_size, roi_mask, cutoff=neg_cutoff):
            patch = img[y :y + patch_size, 
                        x :x + patch_size]
            
            bkg_bbs.append((x, y, patch_size, patch_size))
            patch_img = Image.fromarray(patch.astype('int32'), mode='I')
            filename =f"{basename}_{sampled_bkg:04d}_img.png"
            fullname = bkg_dir / filename
            patch_img.save(str(fullname))
                
            if save_patch_mask:
                patch_mask = roi_mask[y:y + patch_size,
                                      x:x + patch_size]
                patch_mask_img = Image.fromarray(patch_mask.astype('uint8'))
                filename = f"{basename}_{sampled_bkg:04d}_mask.png"
                fullname = bkg_dir / filename
                patch_mask_img.save(str(fullname))
                
            sampled_bkg += 1
            if verbose:
                print(f"{filename}: sampled a bkg at=", (x,y))
                
    return abn_bbs, bkg_bbs
                

def sample_hard_negatives(img:np.array, roi_mask:np.array, out_dir, mask_id:str, 
                        bounding_box: list,   
                        patch_size=256, neg_cutoff=.35, nb_bkg=100, 
                        bkg_dir='background', 
                        save_patch_mask=True, verbose=False):
    """Samples hard negative patches from an image with an abnomality ie. near the ROI but not overlapping more
    than the cutoff. The function samples many patches and accepts only those that do not overlap.

    Args:
        img (np.array): input image
        roi_mask (np.array): input mask
        out_dir (_type_): root output directory
        mask_id (str): Used to name patches and extract pathology and type of lession.
        bounding_box (list): [x, y, w, h]
        patch_size (int, optional): Size of extracted patches. Defaults to 256.
        neg_cutoff (float, optional): maximum allowed overlap. Defaults to .35.
        nb_bkg (int, optional): number of extracted paches per image. Defaults to 100.
        bkg_dir (str, optional): subfolder in output_dir. Defaults to 'background'.
        save_patch_mask (bool, optional): Save binary image with lession. Defaults to True.
        verbose (bool, optional): Print verbose messages. Defaults to False.
    """
    '''WARNING: the definition of hns may be problematic.
    There has been study showing that the context of an ROI is also useful
    for classification.
    '''
    out_dir = pathlib.Path(out_dir)
    
    bkg_dir = out_dir / bkg_dir

    if not bkg_dir.exists():
        bkg_dir.mkdir(parents=True)
    
    
    basename = mask_id.split('/')[-1].replace('.png', '') #D_4124_1.RIGHT_CC_MASS_MALIGNANT_mask_0
    

    rx,ry,rw,rh = round(bounding_box[0]),round(bounding_box[1]), round(bounding_box[2]), round(bounding_box[3])

    assert rw > 0, "rw > 0"
    assert rh > 0, "rh > 0"
    assert rx >= 0, "rx >= 0"
    assert ry >= 0, "ry >= 0"
    assert rx + rw < img.shape[1], f"rx + rw < img.shape[1], {rx + rw} < {img.shape[1]}"
    assert ry + rh < img.shape[0], f"ry + rh < img.shape[0], {ry + rh} < {img.shape[0]}"
    
    
    img = add_img_margins(img, patch_size//2)
    roi_mask = add_img_margins(roi_mask, patch_size//2)


    bkg_bbs = [] # background bounding boxes
    
    # Sample hard negative samples.
    sampled_bkg = 0
    while sampled_bkg < nb_bkg:
        x1,x2 = (rx - patch_size//2, rx + rw + patch_size//2)
        y1,y2 = (ry - patch_size//2, ry + rh + patch_size//2)
        x1 = np.clip(x1,0, img.shape[1] - patch_size)
        x2 = np.clip(x2,0, img.shape[1] - patch_size)
        y1 = np.clip(y1,0, img.shape[0] - patch_size)
        y2 = np.clip(y2,0, img.shape[0] - patch_size)
        x = np.random.randint(x1, x2)
        y = np.random.randint(y1, y2)
        if not overlap_patch_roi((x+patch_size //2,y+patch_size //2), patch_size, roi_mask, cutoff=neg_cutoff):
            patch = img[y :y + patch_size, x :x + patch_size]
            bkg_bbs.append((x, y, patch_size, patch_size))
            patch_img = Image.fromarray(patch.astype('int32'), mode='I')
            filename =f"{basename}_{sampled_bkg:04d}_img.png"
            fullname = bkg_dir / filename
            patch_img.save(str(fullname))

            if save_patch_mask:
                patch_mask = roi_mask[y:y + patch_size, x:x + patch_size]
                patch_mask_img = Image.fromarray(patch_mask.astype('uint8'))
                filename = f"{basename}_{sampled_bkg:04d}_mask.png"
                fullname = bkg_dir / filename
                patch_mask_img.save(str(fullname))            
            
            sampled_bkg += 1
            
            if verbose:
                print(f"{filename}: sampled a bkg at=", (x,y))
    return bkg_bbs

def sample_blob_negatives(img: np.array, roi_mask: np.array , out_dir:str, mask_id:str, 
                        blob_detector, 
                        patch_size=256, neg_cutoff=.35, nb_bkg=100, 
                        bkg_dir='background', 
                        save_patch_mask=True, verbose=False):
    
    """Samples negative patches from an image with or without an abnormality
    The center of the patches are sampled from the blobs detected by the blob_detector.
    The function samples many patches and accepts only those that do not overlap.
    
    This function can be used to sample negative patches from images without an abnormality.
    Just use a blank mask as roi_mask.
    
    Args:
        img (np.array): input image
        roi_mask (np.array): input mask
        out_dir (str): root output directory
        mask_id (str): Used to name patches and extract pathology and type of lesion.
        blob_detector (_type_): blob detector object
        patch_size (int, optional): Size of extracted patches. Defaults to 256.
        neg_cutoff (float, optional): maximum allowed overlap. Defaults to .35.
        nb_bkg (int, optional): number of extracted patches per image. Defaults to 100.
        bkg_dir (str, optional): subfolder in output_dir. Defaults to 'background'.
        save_patch_mask (bool, optional): Save binary image with lesion. Defaults to True.
        verbose (bool, optional): Print verbose messages. Defaults to False.

    Returns:
        list: List of seleted bounding boxes [[x, y, w, h], ...]
    """
    
    out_dir = pathlib.Path(out_dir)
    
    bkg_dir = out_dir / bkg_dir

    if not bkg_dir.exists():
        bkg_dir.mkdir(parents=True)
    
    
    basename = mask_id.split('/')[-1].replace('.png', '') #D_4124_1.RIGHT_CC_MASS_MALIGNANT_mask_0

    img = add_img_margins(img, patch_size//2)
    roi_mask = add_img_margins(roi_mask, patch_size//2)
    # Get ROI bounding box.

    bkg_bbs = [] # background bounding boxes


    # Sample blob negative samples.
    key_pts = blob_detector.detect((img/img.max()*255).astype('uint8'))
    
    key_pts = np.random.permutation(key_pts)
    sampled_bkg = 0
    for kp in key_pts:
        if sampled_bkg >= nb_bkg:
            break
        x,y = int(kp.pt[0]), int(kp.pt[1])
        if not overlap_patch_roi((x,y), patch_size, roi_mask, cutoff=neg_cutoff):
            patch = img[y - patch_size//2:y + patch_size//2, 
                        x - patch_size//2:x + patch_size//2]
            bkg_bbs.append((x, y, patch_size, patch_size))
            patch_img = Image.fromarray(patch.astype('int32'), mode='I')
            filename =f"{basename}_{sampled_bkg:04d}_img.png"
            fullname = bkg_dir / filename
            patch_img.save(str(fullname))
            
            if save_patch_mask:
                patch_mask = roi_mask[y - patch_size//2:y + patch_size//2,
                                    x - patch_size//2:x + patch_size//2]
                patch_mask_img = Image.fromarray(patch_mask.astype('uint8'))
                filename = f"{basename}_{sampled_bkg:04d}_mask.png"
                fullname = bkg_dir / filename
                patch_mask_img.save(str(fullname))  
            
            if verbose:
                print("sampled a blob patch at", (x,y))
            sampled_bkg += 1
    return bkg_bbs

#### End of function definition ####



def sample_positive_bb(roi_outline, patch_size=256,
                pos_cutoff=.75, 
                nb_abn=1, 
                verbose=False):

    """Sample positive  bounding boxes from an image. 
    Only the img_size and roi_outline are used to sample the bounding boxes. This is useful for 
    images that will be augmented using a geometrical transformation.
    
    Positive patches are randomly sampled and accepted if the overlap with the ROI is over pos_cutoff.
    
    
    Args:
    img: numpy array, (W,H)
    roi_outline: list, the ROI outline. [(x1,y1), (x2,y2), ...]
    patch_size: int, the patch size.
    pos_cutoff: float, the overlap cutoff for positive patches.
    nb_abn: int, the number of abnormal patches.
    verbose: bool, print out debug info.

    Returns:
    abn_bbs: list with abnormal bounding boxes. [[x, y, w, h]]
    """
    

    

    abn_bbs = [] # abnormal bounding boxes 
    
    out_line = np.array(roi_outline)
    if verbose:
        print("Number of points in outline=", out_line.shape[0])

    xmin = round(out_line[:,0].min())
    xmax = round(out_line[:,0].max())
    ymin = round(out_line[:,1].min())
    ymax = round(out_line[:,1].max())
    
    
    
    # Sample abnormality first.
    sampled_abn = 0
    nb_try = 0
    while sampled_abn < nb_abn:
        x = np.random.randint(xmin, xmax)
        y = np.random.randint(ymin, ymax)
        nb_try += 1
        if nb_try >= 1000:
            print("Nb of trials reached maximum, decrease overlap cutoff by 0.05")
            sys.stdout.flush()
            pos_cutoff -= .05
            nb_try = 0
            if pos_cutoff <= .0:
                raise Exception("overlap cutoff becomes non-positive, "
                                "check roi mask input.")
        # import pdb; pdb.set_trace()
        
        if overlap_bb_boundary([x - patch_size // 2, 
                                y - patch_size // 2, 
                                x + patch_size // 2,
                                y + patch_size // 2],  out_line, cutoff=pos_cutoff):
            
            abn_bbs.append((x, y, patch_size, patch_size))
            
            
            sampled_abn += 1
            nb_try = 0
            if verbose:
                print(f" sampled abn at", (x,y))
    return abn_bbs
                
def sample_negative_bb(image_outline, roi_outline, patch_size=256,
                neg_cutoff=.35, 
                im_size = (4096, 4096),
                nb_bkg=10, 
                verbose=False):

    """Sample negative bounding boxes from an image. 
    Only the img_size and roi_outline are used to sample the bounding boxes. This is useful for 
    images that will be augmented using a geometrical transformation.
    
    Negative patches are randomly sampled and accepted if the overlap with the ROI is below neg_cutoff.
    
    
    Args:
    image_outine: list, the image outline. [(x1,y1), (x2,y2), ...] (4 points), sampling must be inside this outline.
    roi_outline: list, the ROI outline. [(x1,y1), (x2,y2), ...]
    patch_size: int, the patch size.
    neg_cutoff: float, the overlap cutoff for positive patches.
    im_size: tuple, (W,H)
    nb_bkg: int, the number of abnormal patches.
    verbose: bool, print out debug info.

    Returns:
    abn_bbs: list with abnormal bounding boxes. [[x, y, w, h]]
    """
    

    

    abn_bbs = [] # abnormal bounding boxes 
    if roi_outline is not None:
        roi_outline = np.array(roi_outline)
        if verbose:
            print("Number of points in outline=", roi_outline.shape[0])
    else:
        if verbose:
            print("No outline provided, sampling from the whole image")

    # bounding box that encapulates the image outline.
    xmin = np.min(image_outline[:,0]) 
    xmax = np.max(image_outline[:,0])
    ymin = np.min(image_outline[:,1])
    ymax = np.max(image_outline[:,1])
    
    image_polygon = Polygon(image_outline)
    
    # Sample abnormality first.
    sampled_bkg = 0
    nb_try = 0
    while sampled_bkg < nb_bkg:
        x = np.random.randint(xmin, xmax)
        y = np.random.randint(ymin, ymax)
        nb_try += 1
        if nb_try >= 1000:
            print("Nb of trials reached maximum, increase overlap cutoff by 0.05")
            neg_cutoff += .05
            nb_try = 0
            if neg_cutoff > 1.0:
                raise Exception("overlap cutoff becomes non-positive, "
                                "check roi mask input.")
        # import pdb; pdb.set_trace()
        
        if  point_in_polygon([x,y], image_polygon) and (roi_outline is  None or not overlap_bb_boundary([x - patch_size // 2,
                                y - patch_size // 2, 
                                x + patch_size // 2,
                                y + patch_size // 2],  roi_outline, cutoff=neg_cutoff)):
            
            abn_bbs.append((x, y, patch_size, patch_size))
            
            
            sampled_bkg += 1
            nb_try = 0
            if verbose:
                print(f" sampled abn at", (x,y))
    return abn_bbs


def sample_hard_negative_bb(roi_outline, patch_size=256,
                neg_cutoff=.75, 
                nb_bkg=100, 
                verbose=False):

    """Sample hard negatives  bounding boxes from an image. 
    Only the img_size and roi_outline are used to sample the bounding boxes. This is useful for 
    images that will be augmented using a geometrical transformation.
    
    Negatives patches are randomly sampled and accepted if the overlap with the ROI is below neg_cutoff.
    The difference with previous function is that the patches are sampled near the ROI but not overlapping.
    
    
    Args:
    img: numpy array, (W,H)
    roi_outline: list, the ROI outline. [(x1,y1), (x2,y2), ...]
    patch_size: int, the patch size.
    pos_cutoff: float, the overlap cutoff for positive patches.
    nb_abn: int, the number of abnormal patches.
    verbose: bool, print out debug info.

    Returns:
    abn_bbs: list with abnormal bounding boxes. [[x, y, w, h]]
    """
    

    

    bkg_bbs = [] # abnormal bounding boxes 
    
    out_line = np.array(roi_outline)
    if verbose:
        print("Number of points in outline=", out_line.shape[0])

    xmin = round(out_line[:,0].min() - patch_size//2)
    xmax = round(out_line[:,0].max() + patch_size//2)
    ymin = round(out_line[:,1].min() - patch_size//2)
    ymax = round(out_line[:,1].max() + patch_size//2)
    
    W = xmax - xmin
    H = ymax - ymin
    
    
    # Sample abnormality first.
    sampled_bkg = 0
    nb_try = 0
    while sampled_bkg < nb_bkg:
        x = np.random.randint(xmin, xmax)
        y = np.random.randint(ymin, ymax)
        nb_try += 1
        if nb_try >= 1000:
            print("Nb of trials reached maximum, Increasing search sides  0.1xW")
            sys.stdout.flush()
            xmin = xmin - W//10
            xmax = xmax + W//10
            ymin = ymin - H//10
            ymax = ymax + H//10

            nb_try = 0
        
        if not overlap_bb_boundary([x - patch_size // 2, 
                                y - patch_size // 2, 
                                x + patch_size // 2,
                                y + patch_size // 2],  out_line, cutoff=neg_cutoff):
            
            bkg_bbs.append((x, y, patch_size, patch_size))
            
            
            sampled_bkg += 1
            nb_try = 0
            if verbose:
                print(f" sampled abn at", (x,y))
    return bkg_bbs
                

def sample_blob_negative_bb(key_pts,roi_outline, 
                patch_size=256,
                neg_cutoff=.35, 
                nb_bkg=100, 
                verbose=False):

    """Sample positive  bounding boxes from an image. 
    Only the  and roi_outline are used to sample the bounding boxes. This is useful for 
    images that will be augmented using a geometrical transformation.
    
    Negative patches are randomly sampled at blob keypoints and accepted 
    if the overlap with the ROI is below neg_cutoff.
    
    
    Args:
    img: numpy array, (W,H)
    roi_outline: list, the ROI outline. [(x1,y1), (x2,y2), ...]
    patch_size: int, the patch size.
    neg_cutoff: float, the overlap cutoff for positive patches.
    nb_bkg: int, the number of abnormal patches.
    verbose: bool, print out debug info.

    Returns:
    abn_bbs: list with abnormal bounding boxes. [[x, y, w, h]]
    """
    
    bkg_bbs = [] # abnormal bounding boxes 
    
    if roi_outline is not None:
        roi_outline = np.array(roi_outline)
        if verbose:
            print("Number of points in outline=", roi_outline.shape[0])
    else:
        if verbose:
            print("No outline provided, sampling from the whole image at blob keypoints")

    key_pts = np.random.permutation(key_pts)
    
    # Sample at key_pts.
    sampled_bkg = 0
    for kp in key_pts:
        if sampled_bkg >= nb_bkg:
            break
        x,y = kp[0], kp[1]
        
        if roi_outline is None or not overlap_bb_boundary([x - patch_size // 2, 
                                y - patch_size // 2, 
                                x + patch_size // 2,
                                y + patch_size // 2],  roi_outline, cutoff=neg_cutoff):
            
            bkg_bbs.append((x, y, patch_size, patch_size))
            
            
            sampled_bkg += 1
            if verbose:
                print(f" sampled abn at", (x,y))
        else:
            if verbose:
                print(f" rejected abn at", (x,y))
    return bkg_bbs

