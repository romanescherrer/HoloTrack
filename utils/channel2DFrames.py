import numpy as np
from PIL import Image
import pandas as pd
import os
from .image import crop, norm
from .fourier import phasorFFT,FT2Dc,IFT2Dc,Propagator

def findDC(y0,N):
    ''' compute the displacement profile on a NxN image
    ____
    y0 : y position of the plankton
    N : image size
    '''
    pixelsize = 1.12e-6
    u0 = 5*pixelsize
    a = N//2*pixelsize # m
    v = -100*((y0*pixelsize)**2-a**2)+u0
    return(v/pixelsize)


def selectPlancton(k,nc,path,cmap):
    '''
    k : number of classes to select
    nc : number of classes in the dataset
    path : main path eg '.dataset/train/'
    cmap : dict that map the classes ID with the plankton names
    '''
    # selectionner k folder dans path
    folders = os.listdir(path)
    ids = np.random.randint(0,nc,size=k)
    IM , CL = [], []
    for i in ids :
        f = folders[i]
        listing = os.listdir(os.path.join(path, f))
        # prendre 1 seul plancton
        name = listing[np.random.randint(0,len(listing))]


        I = np.array(Image.open(os.path.join(path,f,name)))
        I = crop(I)/255
        IM.append(I)

        CL.append(cmap[f])
    return(IM,CL)

def traj(y0,vmap,n):
    ''' return the x-trajectory of a plankton based on its initil y-position
    y0 : initial y-position (at frame 0)
    vmap : dict that matches the y-position with a x-displacement per frame
    n : number of frames
    '''
    x = np.zeros(n)
    dc = vmap[y0]
    x[0]=0
    for i in range(1,n):
        x[i]=x[i-1]+dc
    return(x)

def GenPosition(Imgs,N,WB=[]):

    C = np.ones((N,N))
    if len(WB)>0 :
        a = WB[0:N,0:N].copy()

    else :
        a = np.ones((N,N))

    IS = []
    Y1 = []
    W1 = []
    H1 = []

    for i in range(0,len(Imgs)):
        m = Imgs[i].shape[0]# hauteur
        zone = a[:,0:Imgs[i].shape[1]].copy()
        prit = np.where(zone==0)[0]
        librey = np.setdiff1d(np.arange(0,512),prit)
        v=[librey[i+1]-librey[i] for i in range(0,len(librey)-1)]
        gaps = np.concatenate((np.array([0]),np.where(np.array(v)!=1)[0],np.array([len(librey)-2])))

        rep = True
        it=0
        itermax = len(gaps)
        while (rep | it<itermax):
            for j in range(0,len(gaps)-1):
                dif = gaps[j+1]-gaps[j]
                if dif> m :
                    if librey[gaps[j+1]]-m>librey[gaps[j]+1]:
                        nm = np.random.randint(librey[gaps[j]+1],librey[gaps[j+1]]-m)
                    else :
                        nm = librey[gaps[j]+1]
                    rep = False
                    it +=1
                    break
                else :
                    it +=1
        if not rep :

            C[nm:nm+m,0:0+Imgs[i].shape[1]]=Imgs[i]
            a[nm:nm+m,0:0+Imgs[i].shape[1]]= 0

            IS.append(i)
            Y1.append(nm)
            H1.append(m)
            W1.append(Imgs[i].shape[1])

    return(Y1,W1,H1,IS,C,a)


def createCanvas2D(N,path,names, framesID,nbframes):
    ''' create a arrays of shape 512x1024xnbframes with the consecutive
    canvas before simulation of the holograms

    N : Height size of the Canvas
    path : '.dataset/train/'
    name = list of the plancton names
    framesID : list, IDs where new plankton arrive in the FOV
    nbframes : number of frames to simulate
    '''

    # compute displacement in the 2D channel
    vect = [ findDC(y0,N) for y0 in range(-N//2,N//2)]
    vmap = {i:val for i,val in enumerate(vect)}

    cmap = {v:i for i,v in enumerate(names)}

    Canvas = np.ones((512,2*512,nbframes))
    IM = []
    CL = []
    PX1 = []
    PY1 = []
    F = [] # frame start
    FID = []
    XX1,YY1= [],[]
    WW1,HH1 = [],[]
    XX2,YY2 = [],[]
    CCL = []


    WB = np.ones(Canvas.shape)
    for f in range(0,nbframes):
        if f in framesID:
           Imgs,classes = selectPlancton(2,13,path,cmap)
           Y1,W1,H1,IDs,C,a  =  GenPosition(Imgs,512,[WB[:,:,0] if f==0 else WB[:,:,f-1]][0])
           for jj in range(len(IDs)):
               IM.append(Imgs[IDs[jj]])
               CL.append(classes[IDs[jj]])

           for i in range(len(IDs)):
                yy = Y1[i]
                inter = traj(yy+Imgs[i].shape[0]//2,vmap,nbframes)
                posi_x = np.zeros(50)

                if f ==0:
                    posi_x = inter.copy()
                else :
                    posi_x[f:]=inter[:-f]

                PX1.append(posi_x)
                PY1.append(yy*np.ones(posi_x.shape))


        for i in range(len(IM)):
            imcrop = IM[i]
            t = -1*imcrop+1
            imcrop = np.exp(-t)

            posi_x = PX1[i]
            posi_y = PY1[i]

            xx = int(posi_x[f])
            yy = int(posi_y[f])


            Canvas[yy:yy+imcrop.shape[0],xx:xx+imcrop.shape[1],f]=imcrop
            WB[yy:yy+imcrop.shape[0],xx:xx+imcrop.shape[1],f]=0


            F.append(f)
            FID.append(i)
            XX1.append(xx)
            YY1.append(yy)
            XX2.append(xx+imcrop.shape[1])
            YY2.append(yy+imcrop.shape[0])
            WW1.append(imcrop.shape[1])
            HH1.append(imcrop.shape[0])
            CCL.append(CL[i])


    res = pd.DataFrame({'frame':F,'ID':FID,'X1':XX1,'Y1':YY1,'X2':XX2,'Y2':YY2,'W':WW1,'H':HH1,'C':CCL})
    return(Canvas,res)


def simulHolo(Canvas,pixelsize,Lambda,z):
    ''' holo simulation '''
    N,_,n = Canvas.shape # nb de frames

    x =  np.ones((2*N,2*N,n))
    x[N//2:N+N//2,:,:]=Canvas
    x  = norm(x)
    H = np.ones((N,N,n))

    area = 2*N*pixelsize
    prop = Propagator(2*N,Lambda,area,z)
    f = phasorFFT(2*N)

    for i in range(n):
        U =  IFT2Dc(FT2Dc(x[:,:,i],f)*np.conj(prop),f)
        holo = abs(U)**2
        holo = norm(holo[N//2:N+N//2,0:N])
        H[:,:,i] = holo
    return(H)
