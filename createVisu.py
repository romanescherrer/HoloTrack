# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 16:22:44 2022

@author: adm.rscherrer
"""

import matplotlib.pyplot as plt
import numpy as np 
from PIL import Image 
import os 
import glob 
from LNBI.utils.image import crop
path ='dataset/train/'

IM = []
for f in os.listdir(path):
    imgfile = glob.glob(os.path.join(path, f,'*.jpg'))[2]
    print(imgfile)
    I = np.array(Image.open(imgfile))
    IM.append(I)
    
#%%

fig, ax = plt.subplots(2,6,dpi=200)
k = 0
for i in range(2):
    for j in range(6):
        ax[i,j].imshow(crop(IM[k]),'gray')
        ax[i,j].set_xticks([])
        ax[i,j].set_yticks([])
        k+=1
