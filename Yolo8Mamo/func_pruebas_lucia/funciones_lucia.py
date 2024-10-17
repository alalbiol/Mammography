
import sys
import os.path
import pathlib
import numpy as np
import matplotlib.pyplot as plt



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



def create_bbox(overlay):
            
    abnormalities  = list(overlay[2])

    chain_outline = list(abnormalities[0][4][0])
    outline = get_outline_curve(chain_outline)

    maxx = max(outline[0])
    minx = min(outline[0])
    maxy = max(outline[1])
    miny = min(outline[1])

    esquinas = np.array([minx, miny, maxx, maxy])
    return esquinas

def has_overlay(image_path):
    overlay_path = image_path.with_suffix('.OVERLAY')
    return overlay_path.exists()