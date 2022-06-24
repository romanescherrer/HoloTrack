# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 09:49:32 2022

@author: adm.rscherrer
"""

import numpy as np 
from PIL import Image 
import matplotlib.pyplot as plt 
from matplotlib import patches
import os 
import sys
import pandas as pd
sys.path.append('C:\\Users\\adm.rscherrer\\Desktop\\duplicat\\')

from LNBI.utils.image import crop,norm
from LNBI.utils.detection import xywh_xyxy,xyxy_xywh_norm,createTxt
from LNBI.utils.channel2DFrames import   createCanvas2D, simulHolo

from LNBI.utils.plots import imshowgray, imshowpair, plotbbox,plotbboxFormat



#%% 

path = 'dataset/test'
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


nbframes = 50
framesID = [0,5,10,15,25]
N =  512

Canvas,res = createCanvas2D(N, path, names, framesID,nbframes)
Holo = simulHolo(Canvas,pixelsize,Lambda,z)

#%%
ID = 21
imshowpair(Canvas[:N,:N,ID],Holo[:,:,ID])


#%%

Truebboxes =  res[res['frame']==ID][['X1','Y1','X2','Y2']].values
TrueClasses = res[res['frame']==ID]['C'].values


#%%
plotbboxFormat(Holo[:,:,ID],Truebboxes,classes=TrueClasses, format='xyxy',color='k')
#%%


plotbboxFormat(Holo[:,:,ID],Truebboxes, format='xyxy')
plotbboxFormat(Canvas[:N,:N,ID],Truebboxes, format='xyxy')

#%%
from LNBI.utils.yoloutils import getYoloModel
from LNBI.utils.yoloutils import predictOnFrame

w = 'models/Holo/large/best.pt'
model = getYoloModel(w,N)

preds,clsid = predictOnFrame(Holo[:,:,ID],model)

#%%
plotbboxFormat(Holo[:,:,ID],bboxes=preds[:,:-1], classes=clsid,format='xyxy')
plotbboxFormat(Canvas[:N,:N,ID],Truebboxes,classes=TrueClasses, format='xyxy')

#%% test rapide tracking 
from LNBI.utils.trackingkalman import Sort,keepTrajectories,iou_batch,linear_assignment

mot_tracker = Sort( min_hits=3,iou_threshold=0.3)
track_bbx_ids = mot_tracker.update(preds)
coords = track_bbx_ids.tolist()

#%%
plotbboxFormat(Holo[:,:,ID],bboxes=coords, classes=clsid,format='xyxy')


#%% Tracking
mot_tracker = Sort( min_hits=3,iou_threshold=0.3)
# first frame 

preds,clsid = predictOnFrame(Holo[:,:,0],model)
track_bbx_ids = mot_tracker.update(preds)
coords = track_bbx_ids.tolist()
prev_coords = coords.copy()
plotbboxFormat(Holo[:,:,ID],bboxes=preds, classes=clsid,format='xyxy')

TT = keepTrajectories(coords,preds,clsid,1)

for ID in range(1,nbframes):
    preds,clsid = predictOnFrame(Holo[:,:,ID],model)
    track_bbx_ids = mot_tracker.update(preds)
    coords = track_bbx_ids.tolist()
    plotbboxFormat(Holo[:,:,ID],bboxes=preds, classes=clsid,format='xyxy')
    # print(coords)
    
    iou_matrix = iou_batch(preds, np.array(coords)[:,:-1])
    matched_indices = linear_assignment(-iou_matrix)
    CUP =[ np.concatenate((np.array(coords)[matched_indices[i,1],:],clsid[matched_indices[i,0:1]])).tolist()for i in range(matched_indices.shape[0])]
    TT.update(CUP,clsid,ID+1)
    prev_coords = coords.copy()
    
    # plotbboxFormat(Holo[:,:,ID],bboxes=coords, classes=clsid,format='xyxy')



#%%
def trackingFile(TT,nbframe):
    ''' permet de faire un csv des trackers predictions suivant le MOT format
    TT : keepTrajectories() objet qui suit et update les coordonnées des tracker '''
    ids = TT.ids
    F = []
    I = []
    X1,Y1 = [],[]
    WW,HH = [],[]
    BB = [] # for -1

    for i in range(1,nbframe+1): # commence a 1 car mot
        for j,v in enumerate(ids):
            key = '%.1f' %v
            ff = TT.frames[key]
            if i in ff : # si la frame est dans les frames detected
                F.append(i)
                I.append(int(v))

                posi = np.where(np.array(ff)==i)[0][0]

                X1.append(int(TT.trajx[key][posi]))
                Y1.append(int(TT.trajy[key][posi]))
                WW.append(int(TT.trajh[key][posi]))
                HH.append(int(TT.trajw[key][posi]))
                BB.append(int(TT.classID[key][posi]))
    r = pd.DataFrame(data={'frame':F,'ID':I,'X':X1,'Y':Y1,'W':WW,'H':HH,'class':BB})
    return(r)


v = trackingFile(TT,50)
    


#%%
import seaborn as sns

DS = [9,14,30] # [5+4,10+4,15+4,20+4]

IDS = [15,29,49]
fig,ax = plt.subplots(2,3,dpi=300)
# ax[0].imshow(Canvas[0:512,0:512,i],'gray')
for k,i in enumerate(IDS) :
    # fig,ax = plt.subplots(1,2,dpi=300)
    ax[0,k].imshow(Holo[0:512,0:512,i],'gray')
    ax[0,k].set_xticks([])
    ax[0,k].set_yticks([])
    ax[0,k].set_title('frame %i'%(i+1))
    nb = v[v['frame']<=i].groupby('ID').agg({'class':'unique'}).values.astype('float32').ravel().astype(int)
    cpd = pd.DataFrame(data={'c':nb})
    sns.countplot(x='c',data=cpd,ax = ax[1,k],color='w',edgecolor='k')
    ax[1,k].set_ylim(0,5)
    print(nb)
    sub = v[v['frame']==i]
    for j in range(len(sub)):

        ID = sub['ID'].values[j]
        cl =  sub['class'].values[j]
        # si en xyxy
        x1 = float( sub['X'].values[j])
        y1 = float( sub['Y'].values[j])

        # x2 = float( sub['ID'].values[j])
        # y2 = float(one[3])

        w = float( sub['H'].values[j])
        h = float( sub['W'].values[j])


        rect = patches.Rectangle(( x1,y1), w, h, linewidth=0.5, edgecolor='k', facecolor='none')
        ax[0,k].add_patch(rect)
        if x1<512:
            # ax[0,k].text(x1,y1, ' '+str(cl),size=8)
            # ax[0,k].text(x1,y1, ' id %i cl %i'%(ID,cl),size=8)
            ax[0,k].text(x1,y1, str(ID),size=8)

ax[0,0].set_ylabel('tracker ID',size=8)
for i,a in enumerate(ax[1]):
    a.set_xlabel('class ID')
    a.set_ylabel('updated counts',size=8)
    if i >0:
        a.set_ylabel('')




c = []
for ids in TT.ids :
    ids = str(ids)
    clpred,nb = np.unique(TT.classID[ids],return_counts=True)
    print(ids,clpred,nb)
    if len(nb)>1:
        print(clpred)
        c.append(clpred[ np.where(nb==np.max(nb))[0][0]])
    else :
        c.append(clpred[0].astype(np.int))
    # c.append(TT.classID[ids])
cpd = pd.DataFrame(data={'c':c})
val = np.array(res.groupby(['ID'])['C'].mean())


CLNp = np.zeros(13)
CLNt = np.zeros(13)

for i in range(13):
    # II
    for j in c :
        if j==i :
            CLNp[i] +=1
    for k in val :
        if k==i :
            CLNt[i] +=1




# countcomp = pd.DataFrame({'classID': np.arange(0,13),'pred':CLNp,'true':CLNt})
countcomp = pd.DataFrame({'classID': np.concatenate((np.arange(0,13),np.arange(0,13))),
                          'count':np.concatenate((CLNp,CLNt)),
                          'type':['pred']*13+['true']*13})

fig,ax = plt.subplots()
sns.barplot(x='classID',y='count',data=countcomp,hue='type',ax=ax)