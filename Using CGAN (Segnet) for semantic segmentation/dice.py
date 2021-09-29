#for each channel and overall
import numpy as np
def iou_mean(yt,yp):
    intersection=(yt*yp)
    union=np.add(yt,yp,)-intersection
    iou_mean_1_3=np.mean(np.sum((intersection+1),axis=(1,2))/np.sum((union+1),axis=(1,2)),axis=0)
    iou_mean_all=np.mean(np.sum((intersection+1),axis=(1,2))/np.sum((union+1),axis=(1,2)))
    return iou_mean_all,iou_mean_1_3 
def dice_mean(yt,yp):
    intersection=(yt*yp)
    union=np.add(yt,yp)
    dice_mean_all=np.mean(np.sum((2*intersection+1),axis=(1,2))/np.sum((union+1),axis=(1,2)))
    dice_mean_1_3=np.mean(np.sum((2*intersection+1),axis=(1,2))/np.sum((union+1),axis=(1,2)),axis=0)
    return dice_mean_all, dice_mean_1_3
def iou_meanabs(yt,yp):
    intersection=np.sum(np.logical_and(yt,yp),axis=(1,2))
    union=np.sum(np.logical_or(yt,yp),axis=(1,2))
    ious=np.mean((intersection+1)/(union+1))
    return ious
