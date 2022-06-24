# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 09:35:42 2022

@author: adm.rscherrer
"""
import os 
import numpy as np 
from PIL import Image 
import matplotlib.pyplot as plt 
from LNBI.utils.image import crop,norm
from LNBI.utils.detection import xywh_xyxy,xyxy_xywh_norm,createTxt
#%%

def selectPlancton(k,nc,path,cmap):
    # selectionner k folder dans path 
    folders = os.listdir(path)
    ids = np.random.randint(0,nc,size=k)
    IM , CL = [], []
    for i in ids :
        f = folders[i]
        listing = os.listdir(os.path.join(path, f))
        # prendre 1 seul plancton
        name = listing[np.random.randint(0,len(listing))]
        print(f,cmap[f],name)
        
        I = np.array(Image.open(os.path.join(path,f,name)))
        I = crop(I)/255
        IM.append(I)
        
        CL.append(cmap[f])
    return(IM,CL)


def generatePositions(Imgs,N,itermax=20):
    ''' generate first coord so that the images do not overlap
    Returns : 
        Coor : array,  [[xc,yc,w,h],...] in yolo format
        IDS : list of int, ids of the images in Imgs
    '''
    
    Coor = []
    IDS = []
    a = np.ones((N,N))
    for ID in range(len(Imgs)):
        imcrop = Imgs[ID]
        h,w = imcrop.shape 
        
        # NO overlaping condition 
        p = np.where(a==1)
        idx = np.random.randint(len(p[0])) # find empty space 
        posi_y = p[0][idx]
        posi_x = p[1][idx]
        b = np.ones((N,N))
        b[posi_y:posi_y+h,posi_x:posi_x+w] = 0
        c  = a.astype(bool)|b.astype(bool) # if c==False overlapping objects 

        k = 0
        rep = False
        while (posi_y+h>N) | (posi_x+w>N) | (len(np.where(c==0)[0])>0) :
            if k >itermax:
                rep = True

                break
            k +=1
            idx = np.random.randint(len(p[0]))
            posi_y = p[0][idx]
            posi_x = p[1][idx]
            b = np.ones((N,N))
            b[posi_y:posi_y+h,posi_x:posi_x+w] = 0
            c  = a.astype(bool)|b.astype(bool)
            c = c.astype(int)
        # end overlaping condition
        
        if rep!=True:
            c  = a.astype(bool)|b.astype(bool) # update binary mask
            a[posi_y:posi_y+h,posi_x:posi_x+w]=0

            xnorm,ynorm, wnorm, hnorm = xyxy_xywh_norm(posi_x,posi_y,posi_x+w,posi_y+h,N)
            Coor.append([xnorm,ynorm, wnorm, hnorm])
            IDS.append(ID)
    return(Coor,IDS)

def traj(y0,N,vmap,n):
    
    x = np.zeros(n)
    dc = vmap[y0]
    x[0]=0
    for i in range(1,n):
        x[i]=x[i-1]+dc
    return(x)


def findDC(y0,N):
    '''
    vitesse en um/s 
    a : distance entre les deux plans en um 
    x0 : unite en um
    pixelsize = 1.12 um 
    
    '''
    pixelsize = 1.12e-6
    u0 = 5*pixelsize
    a = N//2*pixelsize # m
    v = -100*((y0*pixelsize)**2-a**2)+u0
    return(v/pixelsize)

#%%
names= ['hydromedusae_solmaris',
 'P16',
 'chaetognath_sagitta',
 'trichodesmium_puff',
 'echinoderm_larva_seastar_brachiolaria',
 'diatom_chain_tube',
 'P17',
 'acantharia_protist',
 'copepod_calanoid',
 'appendicularian_s_shape',
 'P1',
 'copepod_cyclopoid_oithona',
 'protist_noctiluca']
Lambda = 520e-9
pixelsize = 1.12e-6
z  = 0.8e-3
dz = 10e-9

vect = [ findDC(y0,512) for y0 in range(-512//2,512//2)]
vmap = {i:val for i,val in enumerate(vect)}
nbframes =50