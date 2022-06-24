# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 19:29:39 2022

@author: adm.rscherrer
"""

import os 
import numpy as np 
from PIL import Image 
import sys 
from pathlib import Path
import matplotlib.pyplot as plt 
import glob 
from matplotlib import patches 
# sys.path.append('C:\\Users\\adm.rscherrer\\Desktop\\duplicat\\')

from LNBI.utils.image import crop,norm
from LNBI.utils.plots import imshowgray, imshowpair, plotbbox
from LNBI.utils.fourier import phasorFFT,IFT2Dc,FT2Dc,Propagator
from LNBI.utils.detection import xywh_xyxy,xyxy_xywh_norm,createTxt

from yolov5.utils.torch_utils import select_device
from yolov5.utils.general import non_max_suppression
from yolov5.models.experimental import attempt_load
import torch

#%%
def nextBBox(Coor,N,n):
    ''' create a random trajectory
    Coor - array, [[xc,yc,w,h],..] yolo format
    N - Int, used to denorm Coor
    n - int, number of frame to simulate
    
    '''
    x0 = Coor[0]*N
    y0 = Coor[1]*N

    x = np.ones(n)*x0 + np.arange(0,n)
    y = np.ones(n)*y0
    return(x,y)
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

def createSeqCanvas(Imgs,Classes,N,itermax,n=10):
    '''
    Parameters
    ----------
    Imgs : list 
        list de ROI en grayscale (0,1)
    N : int
        image size

    Returns
    -------
    None.

    '''
    # initialiser les positions

    Coor,IDs =  generatePositions(Imgs,N,itermax)
    CL = Classes[IDs]
    Canvas = np.ones((2*N,2*N,n))
    
    PX,PY = [],[]
    
    
    for i,k in enumerate(IDs) :
        imcrop = Imgs[k]
        t = -1*imcrop+1
        imcrop = np.exp(-t)
        coor = Coor[i]
        
        posi_x, posi_y = nextBBox(coor,N,n)
        PX.append(posi_x)
        PY.append(posi_y)
        
        
        for j in range(n):
            
            xx = int(posi_x[j])
            yy = int(posi_y[j])
            Canvas[yy:yy+imcrop.shape[0],xx:xx+imcrop.shape[1],j]=imcrop
    
    
    return(Canvas,Coor,CL,PX,PY)

def simulationHoloBatch(Canvas,pixelsize,Lambda,z):
    '''
    ______
    z : float   distance en metre
    Lambda = 520e-9
    pixelsize = 1.12e-6
    '''


    # padding l'image pour la simulation
    N = Canvas.shape[0]
    n = Canvas.shape[-1]
    x = np.ones((2*N,2*N,n))
    M = round(N/2)
    x[M:-M,M:-M,:]= Canvas
    x = norm(x)
    # holo 
    
    H = np.zeros(Canvas.shape)

    # donnee pour les fft
    area = len(x)*pixelsize
    prop = Propagator(2*N,Lambda,area,z)
    f = phasorFFT(2*N)
    
    for i in range(n): # pour chaque frame 
        # simulation de l onde
        U =  IFT2Dc(FT2Dc(x[:,:,i],f)*np.conj(prop),f)
        holo = abs(U)**2

        holo = norm(holo[M:-M,M:-M])

        H[:,:,i] = holo
    return(H)
#%% change dir 
path= 'C:/Users/adm.rscherrer/Desktop/duplicat/LNBI/dataset/train'
folders = os.listdir(path)
nc = len(folders)


IM = []
NA = []
for f in folders : 
    name = glob.glob(os.path.join(path,f,'*.jpg'))[0]
    NA.append(name)
    
    # I = np.array(Image.open(os.path.join(f,name)))
    I = np.array(Image.open(name))
    I = crop(I)/255
    IM.append(I)
    
#%% test rapide

N = 256
n = 50  # frames 
Imgs = IM[0:4]
Classes = np.arange(0,4)
Canvas,C,CL,PX,PY=createSeqCanvas(Imgs=Imgs,Classes=Classes,N=N,itermax=10,n=n)
#%% simulate holo from Canvas 
Lambda = 520e-9
pixelsize = 1.12e-6
z  = 0.8e-3
dz = 10e-9

H = simulationHoloBatch(Canvas,pixelsize,Lambda,z)


#%% load model 
conf_thres=0.5
iou_thres=0.45
max_det=1000
agnostic_nms=False
classes=None

# weights = 'C:/Users/adm.rscherrer/Desktop/Conference/scenario1/Holo/runs/train/smallCanvas_8_400/weights/lasttest.pt'
weights = 'C:/Users/adm.rscherrer/Desktop/Conference/scenario1/Holo/runs/train/small_8_400/weights/lasttest.pt'
# params
half=False
augment = False 
visualize = False 

device = select_device('')
half &= device.type != 'cpu' 
w = weights[0] if isinstance(weights, list) else weights
classify = False
model = attempt_load(weights, map_location=device) 
stride = int(model.stride.max()) 

imgz = (512,512)

model(torch.zeros(1, 3, *imgz).to(device).type_as(next(model.parameters())))  # run once
#%% visualize pred on one frame

frame =  H[0:2*N,0:2*N,10]

mu, sigma = 0, 0.01
s = np.random.normal(mu, sigma, size = frame.shape)

frame = norm(s+frame)

a = np.expand_dims(frame ,0)
a = np.concatenate((a,a,a),axis=0)
a = np.expand_dims(a,0)

img = a.astype('float32')
img = torch.from_numpy(img).to(device)

det = model.model[-1]
det.training = False
out,train_out= model(img)
pred =  non_max_suppression(out, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
print(pred)


#%%
preds = pred[0]
fig,ax = plt.subplots(dpi=300)
ax.imshow(frame[0:2*N,0:2*N],'gray')

preds = preds.cpu().detach().numpy()
for i in range(len(preds)):
    one = preds[i]
    cl = int(one[-1])
    
    # si en xyxy
    x1 = float(one[0])
    y1 = float(one[1])
    
    x2 = float(one[2])
    y2 = float(one[3])
    
    w = abs(x1-x2)
    h = abs(y1-y2)

    rect = patches.Rectangle(( x1,y1), w, h, linewidth=1, edgecolor='r', facecolor='none')
    plt.text(x1,y1-5, str(cl))
    ax.add_patch(rect)
ax.set_xticks([])
ax.set_yticks([])


#%%
