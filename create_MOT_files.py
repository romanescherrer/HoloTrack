# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 20:36:24 2022

@author: adm.rscherrer
"""

import os
import sys
import argparse
from pathlib import Path

from tqdm import tqdm
import numpy as np
import torch 

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # add HoloTrack
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

# simulation plankton video 
from HoloTrack.utils.channel2DFrames import createCanvas2D, simulHolo

# tracking with SORT 
from HoloTrack.utils.trackingkalman import Sort,keepTrajectories,iou_batch,linear_assignment

# utils to save txt files for TrackEval 
from HoloTrack.utils import trackevalutils


# plots 
from HoloTrack.utils.plots import plotbboxFormat

def prepare_img(holo):
    holo = np.round(holo*255)
    holo = np.expand_dims(holo,-1)
    holo = holo.astype(np.uint8)
    img = np.concatenate((holo,holo,holo),axis=-1)
    return(img)

def predictOnFrame(holo,model):
    img = prepare_img(holo)
    results = model(img)

    preds = results.xyxy[0].cpu().detach().numpy()
    bbox = preds[:,:5] # xyxy conf format 
    clsid = preds[:,-1] # class id 
    return(bbox,clsid)



def predictionSORT(I,model,visu=False):
    ''' I : ndarray of shape (512,512,nbframes)'''
    nbframes = I.shape[-1]
    
    # ini sort tracker on first frame 
    mot_tracker = Sort( min_hits=3,iou_threshold=0.3)
    preds,clsid = predictOnFrame(I[:,:,0],model)
    # update kalman filters
    track_bbx_ids = mot_tracker.update(preds) 
    # keep track of yolo prediction and SORT
    coords = track_bbx_ids.tolist()
    KT = keepTrajectories(coords,preds,clsid,1)
    
    if visu : 
        plotbboxFormat(I[:,:,0],bboxes=preds, classes=clsid,format='xyxy')
    

    for ID in range(1,nbframes):
        preds,clsid = predictOnFrame(I[:,:,ID],model)
        track_bbx_ids = mot_tracker.update(preds)
        coords = track_bbx_ids.tolist()
        
        # associate yolo preds with kalman filters preds
        iou_matrix = iou_batch(preds, np.array(coords)[:,:-1])
        matched_indices = linear_assignment(-iou_matrix)
        CUP =[ np.concatenate((np.array(coords)[matched_indices[i,1],:],clsid[matched_indices[i,0:1]])).tolist()for i in range(matched_indices.shape[0])]
        #keep track of yolo prediction and SORT
        KT.update(CUP,clsid,ID+1)

        if visu : 
            plotbboxFormat(I[:,:,ID],bboxes=preds, classes=clsid,format='xyxy')
                           
    # process TT and return a dataFrame with the results 
    det = trackevalutils.createDETfile(KT,nbframes)
    
    return(det)
    






def main(opt):
    path = opt.data
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

    framesID = [0,5,10,15,25]
    N =  512

    mot_challenge_name = opt.mot_challenge_name
    nbframes = opt.nbframes
    weights = opt.weights 
    nbsim = opt.nbsim
    print(ROOT)
    print(str(FILE))
    
    # # load yolo model
    pathmodel = os.path.join(str(ROOT),weights)
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=pathmodel,force_reload=opt.force_reload)
    
    for i in tqdm(range(nbsim)):
        name = 'I%.3i'%i
        # simulate 1 video with nbframes that contains plankton that are moving in a 2D channel
        Canvas,res = createCanvas2D(N, path, names, framesID,nbframes)
        Holo = simulHolo(Canvas,pixelsize,Lambda,z)
        
        # detect and track with Yolo & SORT 
        det = predictionSORT(Holo,model, visu=False)
        
        # create and save the txt files for the MOT evaluation 
        gtfile = trackevalutils.createGTfile(res,min_hits=3)
        
        
        trackevalutils.saveGTfile(gtfile,str(ROOT)+'/HoloTrack',mot_challenge_name,name,seqlength=nbframes)
        trackevalutils.saveDETfile(det,str(ROOT)+'/HoloTrack',mot_challenge_name,name)

    
    
    

    
def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='dataset/test', help='folder dataset')
    parser.add_argument('--nbsim', type=int, default=2, help='number of video to simulate')
    parser.add_argument('--nbframes', type=int, default=50, help='number of frames per video')
    parser.add_argument('--mot-challenge-name', type=str, default='Holo', help='mot challenge name for trackEval')
    parser.add_argument('--weights', type=str, default='Holotrack/models/Holo/small/best.pt', help='path/to/model.pt')
    parser.add_argument('--force-reload', action='store_true', help='force reload for torch.hub.load')
    
    
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
