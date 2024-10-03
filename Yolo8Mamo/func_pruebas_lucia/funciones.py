import numpy as np
import os
import globc
import subprocess


def read_lossless_dcm_raw(fname):
    """Read raw image data from lossless dcm."""
    #use the standford cmdline tool to decompress
    try:
        mess=subprocess.check_output(['pvrg-dcm','-d','-s',fname],
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