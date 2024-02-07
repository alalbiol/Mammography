from converter_utils import read_lossless_jpeg_raw
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from converter_utils import read_lossless_jpeg_raw, convert_case_npy
import cv2
import shutil
import time



def get_case_from_fn(fn):    
    case_folder_in = pathlib.Path(fn).parent
    case = "/".join(case_folder_in.parts[-3:])
    return case

def npy2png(npy_file, png_file, delete_npy=True):
    """Load image."""
    #load, rescale to uint8 and invert
    #get between 0.8 and 2.9
    #originally scale is from 3.0 to 0?
    im=np.load(npy_file)
    im=3-3*(im/65535.)
    im=np.clip(im,0.8,2.9)-0.8
    im=(255.* im/im.max()).astype(np.uint8)
    
    cv2.imwrite(png_file,im)
    
    #delete npy
    if delete_npy:
        npy_file.unlink()


    # f=S/max(im.shape) #resize to max S!
    # im=cv2.resize(im,(0,0),fx=f,fy=f)
    
    # if flip:  #flip if asked for
    #     im=np.flipud(im)
    # if rot!=0: #rotate if asked for
    #     origo=(im.shape[1]/2.,im.shape[0]/2.)
    #     M = cv2.getRotationMatrix2D(origo,rot,1)
    #     im = cv2.warpAffine(im,M,im.shape[::-1])
    return im



#S=float(sys.argv[1])

#dont redo done images
# done=glob(dout+'*.png')

# #train images
# with open('mammo_devkit/ImageSets/ddsm.txt') as f:
#     for l in f:
#         in_fn='_'.join(l.strip().split('_')[:5])
#         fn=l.strip()
                    
#         print fn,
#         arr_fn = din+in_fn+'.npy'
#         im_fn = dout+fn+'.png'
        
#         if im_fn in done:
#             print 'skipped'
#             continue
    
#         #load im
#         flip = fn.split('_')[-1] =='flipud' #flip it?
#         if fn.split('_')[-1].find('rot')!=-1: #rot it?
#             rot=int(fn.split('_')[-1].split('rot')[1])
#         else:
#             rot=0
#         im=load_im(arr_fn,flip,rot) #load
        
        #save it        
        # print 'added'


class PrepareDDSM(object):
    def __init__(self, ddsm_dir_in, ddsm_dir_out) -> None:
        self.ddsm_dir_in = pathlib.Path(ddsm_dir_in)
        self.ddsm_dir_out = pathlib.Path(ddsm_dir_out)
        
        self.cases = self.get_cases()

    def get_cases(self):

        all_ljpeg_images = list(self.ddsm_dir_in.glob('**/*.LJPEG'))
        print("Number of images: ", len(all_ljpeg_images))
        
        all_cases = list(set([get_case_from_fn(fn) for fn in all_ljpeg_images]))
        
        print("Number of cases: ", len(all_cases))
        
        return all_cases
    
    def convert_images_case(self,case, force_redo = False):
        
        print("Processing case: ", case)
        case_folder_out = self.ddsm_dir_out / case
        case_folder_out.mkdir(parents=True, exist_ok=True)
        
        case_folder_in = self.ddsm_dir_in / case
        
        number_of_jpgs = len(list(case_folder_in.glob('*.LJPEG')))
        number_of_pngs = len(list(case_folder_out.glob('*.png')))
        
        if not force_redo and number_of_jpgs == number_of_pngs:
            print("Case already converted")
            return

        convert_case_npy(case_folder_in, case_folder_out)
        
        npy_files = list(case_folder_out.glob('*.npy'))
                
        for npy_file in npy_files:
            png_file = case_folder_out / (npy_file.stem + '.png')
            npy2png(npy_file, str(png_file), delete_npy=True)
            print("Converted: ", png_file)
            
    def copy_overlays_case(self, case):
        case_folder_out = self.ddsm_dir_out / case
        case_folder_in = self.ddsm_dir_in / case
        
        overlay_files = list(case_folder_in.glob('*.OVERLAY'))
        # copy overlays using shutil
        for overlay_file in overlay_files:
            overlay_file_out = case_folder_out / overlay_file.name
            print("Copying: ", overlay_file_out)
            shutil.copy(overlay_file, overlay_file_out)
            
    def convert_case(self, case, force_redo = True):
        self.convert_images_case(case, force_redo = force_redo)
        self.copy_overlays_case(case)
        
    def convert_all_cases(self, force_redo = True, parallel = False):
        if parallel:
            from joblib import Parallel, delayed
            Parallel(n_jobs=-1)(delayed(self.convert_case)(case, force_redo=force_redo) for case in self.cases)
 
        else:
            for case in self.cases:
                self.convert_case(case, force_redo=force_redo)
    


import argparse

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='Prepare DDSM data')
    parser.add_argument('--ddsm_dir_in', default='/home/alalbiol/Data/mamo/cases', type=str, help='Path to the DDSM data')
    parser.add_argument('--ddsm_dir_out',default='/tmp/ddsm/', type=str, help='Path to the DDSM data')
    
    args = parser.parse_args()
    
    ddsm = PrepareDDSM(args.ddsm_dir_in, args.ddsm_dir_out)
    
    start_time = time.time()
    ddsm.convert_all_cases(force_redo=True, parallel=True)
    
    print("--- %s seconds ---" % (time.time() - start_time))
    
    