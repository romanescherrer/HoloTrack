import numpy as np


def xywhn_xyxy_box(box,N):
    '''convert a box in the xywhn format to xyxy format'''
    xc = box[:,0:1]*N
    yc = box[:,1:2]*N
    w = box[:,2:3]*N
    h = box[:,3:4]*N
    x1 = xc-w/2
    y1 = yc-h/2
    x2 = x1+w
    y2 = y1+h

    xyxy = np.hstack((x1,y1,x2,y2))
    return(xyxy)


def xyxy_xywh_norm(x1,y1,x2,y2,N):
    ''' normalize coords for yolo '''
    w = abs(x1-x2)
    h = abs(y1-y2)
    xc = min(x1,x2)+ w/2
    yc = min(y1,y2)+ h/2

    xc /=N
    yc /=N
    h /=N
    w /=N
    return(xc,yc,w,h)


def xywh_xyxy(x,y,w,h):
    x1 = x-w/2
    x2 = x1+w
    y1 = y-h/2
    y2 = y1+h
    return(x1,y1,x2,y2)



def createTxt(name,Coor,Classe):
    ''' create a txt file in yoloV5 format
    _______
    name : str   path+name of the txt file
    Coor : list  coordinates of the bbox
    Class : list  classes ID of plankton
    '''
    f = open(name,"w+")
    for i in range(len(Coor)):
        f.write('%i ' %(Classe[i]))
        for j in range(4):
            f.write(str(Coor[i][j]))
            f.write(' ')
        f.write('\n')
    f.close()
    return()
