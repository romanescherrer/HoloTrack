import numpy as np


def phasorFFT(N):
    '''
    TF(U2(p,q))=exp(i*pi(p+q))*TF([U1(m,n)*exp(ipi(m+n))])

    '''
    m,n=np.meshgrid(np.arange(0,N),np.arange(0,N))
    f1=np.cos(np.pi*(m+n))+np.zeros((N,N))*1j
    return(f1)

def FT2Dc(X,f1):
    ''' f1=phasorFFT'''
    FT=np.fft.fft2(f1*X)
    out=f1*FT
    return(out)


def IFT2Dc(X,f1):
    FT=np.fft.ifft2(np.conj(f1)*X)
    out=np.conj(f1)*FT
    return(out)

def Propagator(N,Lambda,area,z):
    '''
    Angular spectrum
    parameters :
        N : Image size (must be squared)
        Lambda : wave length [meters]
        area : pixelsize x N
        z : object/cam distance  [meters]
        '''
    xccd,yccd = np.meshgrid(np.arange(0,N),np.arange(0,N))
    alpha = Lambda*(xccd-N/2)/area
    beta = Lambda*(yccd-N/2)/area

    comp = (-2j*np.pi*z*np.sqrt(1-alpha**2-beta**2)/Lambda)
    p=np.exp(comp)
    return(p)
