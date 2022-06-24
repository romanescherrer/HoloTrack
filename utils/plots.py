import matplotlib.pyplot as plt
from matplotlib import patches

def imshowgray(I,dpi=300,title=None):
    fig, ax = plt.subplots(dpi=dpi)
    ax.imshow(I,'gray')
    if title :
        ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

def imshowpair(X,Y,dpi=300,titles=['','']):
    f,ax = plt.subplots(1,2,dpi=dpi)
    ax[0].imshow(X,'gray')
    ax[1].imshow(Y,'gray')
    for i in range(len(ax)):
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    ax[0].set_title(titles[0])
    ax[1].set_title(titles[1])

def plotbbox(I,bboxes,classes=None, dpi=300,title=None,color = 'r'):
    ''' bboxes are in the format xywh normalized (yolov5 format)'''
    fig,ax = plt.subplots(dpi=300)
    ax.imshow(I,'gray')
    N,_= I.shape
    for i in range(len(bboxes)):
        xc = bboxes[i][0]*N
        yc = bboxes[i][1]*N
        w =  bboxes[i][2]*N
        h =  bboxes[i][3]*N
        rect = patches.Rectangle((xc-w/2,yc-h/2), w, h, linewidth=1, edgecolor=color, facecolor='none')
        if classes :
            cl = classes[i]
            plt.text(xc-w//2,yc-h//2, str(cl),color=color)
        ax.add_patch(rect)
    ax.set_xticks([])
    ax.set_yticks([])

def plotbboxFormat(I,bboxes,format = 'xyxy',classes=[], dpi=300,title=None,color = 'r'):
    ''' bboxes are in the format xywh normalized (yolov5 format)'''
    fig,ax = plt.subplots(dpi=300)
    ax.imshow(I,'gray')
    N,_= I.shape
    ax.set_xticks([])
    ax.set_yticks([])


    if format =='xyxy':
        for i in range(len(bboxes)):
            x1 =  bboxes[i][0]
            y1 = bboxes[i][1]
            x2 = bboxes[i][2]
            y2 = bboxes[i][3]
            rect = patches.Rectangle((x1,y1), abs(x2-x1), abs(y2-y1), linewidth=1, edgecolor=color, facecolor='none')
            if len(classes)>0:
                cl = int(classes[i])
                plt.text(x1,y1,str(cl),color=color)
            ax.add_patch(rect)

    elif format =='xywhn':
        for i in range(len(bboxes)):
            xc = bboxes[i][0]*N
            yc = bboxes[i][1]*N
            w =  bboxes[i][2]*N
            h =  bboxes[i][3]*N
            rect = patches.Rectangle((xc-w/2,yc-h/2), w, h, linewidth=1, edgecolor=color, facecolor='none')
            if len(classes)>0 :
                cl = int(classes[i])
                plt.text(xc-w//2,yc-h//2, str(cl),color=color)
            ax.add_patch(rect)
    else :
        for i in range(len(bboxes)):
            xc = bboxes[i][0]
            yc = bboxes[i][1]
            w =  bboxes[i][2]
            h =  bboxes[i][3]
            rect = patches.Rectangle((xc-w/2,yc-h/2), w, h, linewidth=1, edgecolor=color, facecolor='none')
            if len(classes)>0 :
                cl = int(classes[i])
                plt.text(xc-w//2,yc-h//2, str(cl),color=color)
            ax.add_patch(rect)
