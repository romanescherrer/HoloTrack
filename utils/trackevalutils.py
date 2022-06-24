# utils to generate txt files for TrackEval MOT metrics

import numpy as np
import pandas as pd
import os 

def createDETfile(TT,nbframe):
    '''
    creates a dataframe of the YOLO+SORT trackings for 1 video that follows the
    trackEval format :
    <frameID>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
    ------
    TT : keepTrajectories objects
    nbframes = number of frames during the simulation


     '''
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
                BB.append(-1)
    r = pd.DataFrame(data={'frame':F,'ID':I,'X':X1,'Y':Y1,'W':WW,'H':HH,'conf':BB,'x':BB,'y':BB,'z':BB})
    return(r)


def createGTfile(res,min_hits=0):
    '''
    creates a dataframe of the Ground Truth label for the MOT evaluation
     that follows the trackEval format :
    <frameID>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
    ------
    res : dataframe generated during the holograms simulation
    min_hits : additional arg '''
    a = res.copy()
    # remove Bounding box that are not in the Field of View of the holograms
    iddrop = a[a['X1']>=512].index.tolist()
    a = a.drop(index=iddrop)

    iddrop = []

    if min_hits>0: # min_hits before sort detection
        uid = res.ID.unique()[2:]
        for I in uid :
            iddrop += a[a['ID']==I].index.tolist()[0:min_hits]
        a = a.drop(index=iddrop)

    a['frame'] = a['frame']+1 #start at 1 for MOT evaluation
    a['ID'] = a['ID']+1
    # switch for some reason
    w = a['H'].values.tolist()
    h = a['W'].values.tolist()

    a = a.drop(columns=['W','H','X2','Y2','C'])
    a['W'] = w
    a['H'] = h
    a['conf']=len(a)*[-1]
    a['x']=len(a)*[-1]
    a['y']=len(a)*[-1]
    a['z']=len(a)*[-1]
    return(a)

def saveDETfile(det,path, mot_challenge_name,name):
    '''
            PREDS file for TrackEval
    ------------------------------------
    create a folder that containes the MOT preds in the form :

    |MOT_data/trackers/mot_challenge/<mot_challenge_name>//SORT//data//<name.txt>


    One txt file per video is created. The txt file respected the following
    format:

    <frameID>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
    where <x>,<y>,<z> = -1,-1,-1 (2D detection & tracking)


     ------------------------------------

    The folders (gt and trackers) in HoloTrack/MOT_data can be placed directly
    to the TrackEval repo at  :
        trackeval/data/

    '''

    ptrackparent = os.path.join(path,'MOT_data/trackers/mot_challenge/',mot_challenge_name+'-train')
    ptrackers = os.path.join(ptrackparent,'SORT')

    # ptrackers = 'C:\\Users\\adm.rscherrer\\Desktop\\TrackEval-master\\TrackEval-master\\trackeval\\data\\trackers\\mot_challenge\\'+mot_challenge_name+'-train\\'+det_name+'\\data\\'

    if not os.path.exists(ptrackparent):
        os.mkdir(ptrackparent)
    if not os.path.exists(ptrackers):
        os.mkdir(ptrackers)
        os.mkdir(os.path.join(ptrackers,'data'))

    det.to_csv(os.path.join(ptrackers,'data',name+'.txt'),header=False,index=False)


def write_seqinfo(name,seqlength,pathsave):
    '''
    write the seqinfo.ini for TrackEval
    '''
    infos = {'name':name,'imDir':'img1','frameRate':50,'seqLength':seqlength,'imWidth':512,'imHeight':512,'imExt':'.jpg'}
    f = open(os.path.join(pathsave,'seqinfo.ini'),"w+")
    lignes = [v+'='+str(infos[v]) for i,v in enumerate(infos)]
    # print(lignes)

    f.write('[Sequence]\n')
    for i in range(len(lignes)):
        f.write(lignes[i])
        f.write('\n')

    f.close()

def saveGTfile(gt,path, mot_challenge_name,name,seqlength):

    '''
    Save the Ground truth labels for MOT evaluation
    ___________________________________
    trackeval/data/gt/mot_challenge/
    |seqmaps
        |<mot_challenge_name>-all.txt
            name
            .
            .
            IM0
        |<mot_challenge_name>-train.txt


    |<mot_challenge_name>-train
        |<name>
            |gt
                |gt.txt
            |seqinfo.ini

    gt : dataframe

    name : forlder name in mot_challenge_name-train
    mot_challenge_name : folder name in seqmaps
    '''

    # save seqmaps
    pseq = os.path.join(path,'MOT_data/gt/mot_challenge/seqmaps')

    if  os.path.exists(os.path.join(pseq,mot_challenge_name+'-all.txt')):
        # if seqmaps exists, just append the file name
        f = open(os.path.join(pseq,mot_challenge_name+'-all.txt'),'a')
        f.write('\n'+name)
        f.close()
    else :
        # else create the seqmaps file that start with the line 'name\n'
        f = open(os.path.join(pseq,mot_challenge_name+'-all.txt'),'w')
        f.write('name\n'+name)
        f.close()
    # do the same with the file <mot_challenge_name>+'train.txt' file
    if  os.path.exists(os.path.join(pseq,mot_challenge_name+'-train.txt')):
        f = open(os.path.join(pseq,mot_challenge_name+'-train.txt'),'a')
        f.write('\n'+name)
        f.close()
    else :
        f = open(os.path.join(pseq,mot_challenge_name+'-train.txt'),'w')
        f.write('name\n'+name)
        f.close()
    # save gt bounding boxes txt
    pgtparent = os.path.join(path,'MOT_data/gt/mot_challenge/',mot_challenge_name+'-train')
    if not os.path.exists(pgtparent):
        os.mkdir(pgtparent)

    os.mkdir(os.path.join(pgtparent,name))
    os.mkdir(os.path.join(pgtparent,name,'gt'))
    # write seqinfo.ini
    write_seqinfo(name,seqlength,os.path.join(pgtparent,name))
    # save gt.txt
    gt.to_csv(os.path.join(pgtparent,name,'gt','gt.txt'),header=False,index=False)
    return()
