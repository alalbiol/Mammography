
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



def create_bbox(overlay):
            
    abnormalities  = list(overlay[2])

    chain_outline = list(abnormalities[0][4][0])
    outline = get_outline_curve(chain_outline)

    maxx = max(outline[0])
    minx = min(outline[0])
    maxy = max(outline[1])
    miny = min(outline[1])

    lado1 = np.array([maxx-minx, 0])  # Vector para el ancho (horizontal)
    lado2 = np.array([0, maxy-miny])  # Vector para la altura (vertical)
    origen = np.array([minx, miny])  # Origen del rectángulo

    # Calcular las otras tres esquinas del rectángulo
    esquina1 = origen
    esquina2 = origen + lado1
    esquina3 = origen + lado1 + lado2
    esquina4 = origen + lado2

    esquinas = np.array([esquina1, esquina2, esquina3, esquina4])
    return esquinas
