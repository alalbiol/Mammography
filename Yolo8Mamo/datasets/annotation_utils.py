import os.path
import numpy as np


DDSM_DIR = './'
def get_annot(im_id,ov_dir=DDSM_DIR+'overlays/'):
    """Return the first overlay curve for each image."""
    ov_fn,ljpeg_fn,ics_fn=get_fns(im_id) #get the filenames
    if not os.path.isfile(ov_dir+ov_fn): 
        return [] #return with empty list if no annotation
    w,h=imshape(ics_fn,ljpeg_fn) #get image hw
    _,_,overlays=read_overlay(ov_dir+ov_fn) #read the overlay file
    annots=[]
    for _,ass,subt,patho,outlines in overlays:
        #only use the first annotation for each image???
        x,y=get_outline_curve(outlines[0])
        annots.append(Annot(im_id,x,y,ass,patho,w,h))
    return annots

def get_fns(im_id):
    """Get overlay ljpeg ics image names from my image id."""
    base_fn=im_id[:8]+'.'+im_id[9:]
    ov_fn=base_fn+'.OVERLAY'
    ljpeg_fn=base_fn+'.LJPEG'
    ics_fn=base_fn.split('.')[0].replace('_','-')+'.ics'
    return ov_fn,ljpeg_fn,ics_fn

def read_overlay(fn):
    """Read ovelay file."""
    with open(fn) as f:
        outlines,l_type,ass,subt,patho,n_outl=[],[],[],[],[],[]
        line = f.readline()
        while line:        
            if line.find('TOTAL_ABNORMALITIES')!=-1:
                n_abn=int(line.split(' ')[1])
            elif line.find('ABNORMALITY')!=-1:
                outlines.append([])
            elif line.find('LESION_TYPE')!=-1:
                l_type.append(line.split(' ')[1].strip())
            elif line.find('ASSESSMENT')!=-1:
                ass.append(int(line.split(' ')[1]))
            elif line.find('SUBTLETY')!=-1:
                subt.append(int(line.split(' ')[1]))
            elif line.find('PATHOLOGY')!=-1:
                patho.append(line.split(' ')[1].strip())
            elif line.find('TOTAL_OUTLINES')!=-1:
                n_outl.append(int(line.split(' ')[1]))
            elif line.find('BOUNDARY')!=-1:
                line=f.readline()
                outline_str=list(map(int,line.strip().split()[:-1]))
                outlines[-1].append(outline_str)
            line=f.readline()
    
    return n_abn,n_outl,list(zip(l_type,ass,subt,patho,outlines))

def get_outline_curve(outline):
    """Get outline as a curve from step format."""
    outline=np.array(outline)
    x,y=outline[:2]
    
    dx_p=(outline==1) | (outline==2) | (outline==3)
    dx_m=(outline==7) | (outline==6) | (outline==5)
    xpos=x+np.cumsum(np.int32(dx_p)-np.int32(dx_m))
    
    dy_p=(outline==5) | (outline==4) | (outline==3)
    dy_m=(outline==7) | (outline==0) | (outline==1)
    ypos=y+np.cumsum(np.int32(dy_p)-np.int32(dy_m))
    
    return xpos,ypos


def imshape(ics_fname,fname):
    """Shape the data into image."""
    #get which image it is, the ics file has info on all 4
    im_type=os.path.basename(fname).split('.')[1]
    
    #get image height
    with open(DDSM_DIR+'icss/'+ics_fname) as f:
    # loop over lines in ics file
        for line in f:
            #look for line with info about this image
            if len(line.split())>0 and line.split()[0]==im_type:
                h=int(line.split()[2])
                w=int(line.split()[4])
    return w,h




def parse_annots(im_id_list):
    """Parse all annotations."""
    annots=[]
    for im_id in im_id_list:
        annots+=get_annot(im_id)
    for a in annots:
        a.resize(S)
    return annots

def load_im_ids():
    """Load the image ids."""
    fns=os.listdir(DDSM_DIR+'clean_data_new/')
    im_id_list=[x[:-4] for x in fns]

    #this image has annotation here but its bout black nothing
    #and there is nothing on the internet about this annotation
    # the thumbnal image has no annotation
    mistakes=['A_1045_1_RIGHT_MLO']
    for m in mistakes:
        if m in im_id_list:
            im_id_list.remove(m)
            
    return im_id_list
    
def load_im_from_a(a,flip=False,rot=0):
    """Load an image from the annotation."""
    din=DDSM_DIR+'clean_data_new/'
    arr_fn = din+a.im_id+'.npy'
    im=load_im(arr_fn,flip,rot)
    return im

def load_im(fn,flip,rot):
    """Load image."""
    #load, rescale to uint8 and invert
    #get between 0.8 and 2.9
    #originally scale is from 3.0 to 0?
    im=np.load(fn)
    im=3-3*(im/65535.)
    im=np.clip(im,0.8,2.9)-0.8
    im=(255.* im/im.max()).astype(np.uint8)
  
    f=S/max(im.shape) #resize to max S!
    im=cv2.resize(im,(0,0),fx=f,fy=f)
    
    if flip:  #flip if asked for
        im=np.flipud(im)
    if rot!=0: #rotate if asked for
        origo=(im.shape[1]/2.,im.shape[0]/2.)
        M = cv2.getRotationMatrix2D(origo,rot,1)
        im = cv2.warpAffine(im,M,im.shape[::-1])
    return im

def plot_annot(im,a):
    """Plot image with annotation."""
    f,ax=plt.subplots(figsize=(9,9))
    plt.imshow(im,cmap='gray')
    plt.plot(a.x,a.y)
    xlim(0,im.shape[1])
    ylim(0,im.shape[0])
    
    bb=a.bb()
    ax.add_patch(
        plt.Rectangle((bb[0], bb[2]),
                       bb[1] - bb[0],
                       bb[3] - bb[2], fill=False,
                       edgecolor='red', linewidth=1.5))
    
    
    
def get_BB_from_outline(outline):
    x = outline[0]
    y = outline[1]
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)
    centerx = (xmin + xmax)/2
    centery = (ymin + ymax)/2
    width = xmax - xmin +1
    height = ymax - ymin +1
    
    return [xmin, ymin, xmax, ymax, centerx, centery, width, height]
    

def read_annotation_image(overlay_file):
    overlay = read_overlay(str(overlay_file))
    #total #abormalities, #oulines in each abnormality, abnormality (see bellow)
    #only read first outline
    abnormalities_in = overlay[2]
    
    abnormalities_out = []
    
    any_malignant = False
    
    for abnormality in abnormalities_in:
        abnormality = list(abnormality)
        chain_outline = list(abnormality[4][0])
        outline = get_outline_curve(chain_outline)
        pathology = abnormality[3]
        if pathology == 'MALIGNANT':
            any_malignant = True
        abnormality[4] = outline
        #l_type,ass,subt,patho,outlines
        abnormality_dict = {'type': abnormality[0],
                            'assessment': abnormality[1],
                            'subtlety': abnormality[2],
                            'pathology': pathology,
                            'outline': outline,
                            'bounding_box': get_BB_from_outline(outline)
                            }
                            
        abnormalities_out.append(abnormality_dict)
    
    for ab in abnormalities_out:
        ab['breast_malignant']=any_malignant  # label breast malignant useful por unproven cases 
    
    
    return abnormalities_out
