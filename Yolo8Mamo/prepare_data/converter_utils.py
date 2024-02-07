"""

Util functions to convert images from the lossless jpeg format.

- Convert all images in a case.
- Use pvrg coverter to decompress lossless jpeg.
- Read raw image data and shape it from the info in the ics files
- Convert pixel values to calibrated optical density
- Save it as numpy binary arrays.

"""

import numpy as np
import os
import glob
import subprocess


def convert_case_npy(case_in_path, case_out_path): 
    """Convert all images in a case."""
    #got to case dir
    #os.chdir(case)
    
    # filter images with these substrings ['LEFT_CC','RIGHT_CC','LEFT_MLO','RIGHT_MLO']
    jpeg_images = [fn for fn in  case_in_path.glob('*.LJPEG') if any(orient in str(fn) for orient in ['LEFT_CC','RIGHT_CC','LEFT_MLO','RIGHT_MLO'])]

    #convert all breast and all views
    
    [convert_image(jpeg_image, case_out_path) for jpeg_image in jpeg_images]


    
def convert_image(jpeg_image, case_out_path):
    """Convert one image."""
    #get filenames
    folder = jpeg_image.parent
    ics_fname= list(folder.glob('*.ics'))[0]
    out_fname = case_out_path / jpeg_image.with_suffix('.npy').name
    
    #load ljpeg image
    im=load_image(jpeg_image,ics_fname)
    #save it
    np.save(out_fname,im)
    

def load_image(fname,ics_fname):
    """ Load a usable ddsm image."""
    #read fromt the lossless jpg
    image_data=read_lossless_jpeg_raw(fname)
    #shape from 1d to image from infor in ics file 
    # the converter mixes up the dimension sometimes
    # but the ics file is always good
    image = shape_image(image_data,ics_fname,fname)

    # map to optical density
    image=map_2_opt_dens(image,ics_fname)
    
    return image


def read_lossless_jpeg_raw(fname):
    """Read raw image data from lossless jpeg."""
    #use the standford cmdline tool to decompress
    try:
        mess=subprocess.check_output(['pvrg-jpeg','-d','-s',fname],
                                stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        # it raises the No trailing marker found! error,
        # but it still does the job
        # should check for error
        mess=e.output
        pass
    
    #read uncompressed image
    fname = str(fname)
    with open(fname+'.1', 'rb') as infile:
        data= np.fromfile(infile, dtype='>u2')
    
    #remove the tmp image file made by the converter
    subprocess.call(['rm',fname+'.1'])
    
    return data


def shape_image(image_data,ics_fname,fname):
    """Shape the data into image."""
    #get which image it is, the ics file has info on all 4
    im_type=os.path.basename(fname).split('.')[1]
    
    #get image height
    with open(ics_fname) as f:
	# loop over lines in ics file
        for line in f:
            #look for line with info about this image
            if len(line.split())>0 and line.split()[0]==im_type:
                h=int(line.split()[2])
    
    #reshape image
    im=np.array(image_data.reshape((h,-1)))
    
    return im


def map_2_opt_dens(image,ics_fname):
    """
    Map pixel values to optical density using the calbiration.
        - http://marathon.csee.usf.edu/Mammography/Database.html#DDSMTABLE
    """
    #get scanner and source
    source,scanner=get_source_and_scanner(ics_fname)

    if source=='A' and scanner=='DBA': 
        #0 pixel values screw up this normalization
        image[image==0]=1
        image = ( np.log10(np.float32(image)) - 4.80662 ) / -1.07553
    elif (source=='B' or source=='C') and scanner=='LUMISYS':
        image = 3.6 - (np.float32(image) - 495) / 1000
    elif source=='A' and scanner=='HOWTEK':
        image = 3.789 - 0.00094568 * np.float32(image)
    elif source=='D' and scanner=='HOWTEK':
        image =3.96604095240593 + (-0.00099055807612) * np.float32(image)
    
    #optical density is between 0 and 3 normally
    # original image has 16 bit resolution
    # storing on float would be a waste of space    
    image = np.uint16( np.clip(image,0,3) * 65535 / 3)

    return image


def get_source_and_scanner(ics_fname):
    """Get the source and the scanner type from the ics file."""
    #read ics file
    with open(ics_fname) as f:
        ics_content=f.readlines()
        ics_content='\n'.join(ics_content)
        
    #get scanner
    scanner = ''
    if ics_content.find(' DBA ')!=-1:
        scanner = 'DBA'
    elif ics_content.find(' LUMISYS ')!=-1:
        scanner = 'LUMISYS'
    elif ics_content.find(' HOWTEK ')!=-1:
        scanner = 'HOWTEK'
        
    #get source site
    source = os.path.basename(ics_fname)[0]
    
    return source,scanner
