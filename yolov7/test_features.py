import torch
import numpy as np
from utils.general import rbox_iou_d2, rbox_iou_shapely
from detectron2.structures import BoxMode, RotatedBoxes, pairwise_iou_rotated

def test_shapely():
    w = 20
    h = 10
    gt = [100, 100, 0.2]
    pt = [100, 100, 0.3]

    gtc = [[gt[0] - w//2, gt[1] - h//2], 
           [gt[0] + w//2, gt[1] - h//2],
           [gt[0] + w//2, gt[1] + h//2],
           [gt[0] - w//2, gt[1] + h//2]]
    
    ptc = [[pt[0] - w//2, pt[1] - h//2], 
            [pt[0] + w//2, pt[1] - h//2],
            [pt[0] + w//2, pt[1] + h//2],
            [pt[0] - w//2, pt[1] + h//2]]

    gtr = []
    ptr = []

    for i in range(4):
        ''' 
        yri=cos(θ)(yi-yc)-sin(θ)(xi-xc)+yc
        xri=sin(θ)(yi-yc)+cos(θ)(xi-xc)+xc
        '''
        gtr.append(-np.sin(gt[2]) * (gtc[i][1] - gt[1]) + np.cos(gt[2]) * (gtc[i][0] - gt[0]) + gt[0]) # append X
        gtr.append(np.cos(gt[2]) * (gtc[i][1] - gt[1]) + np.sin(gt[2]) * (gtc[i][0] - gt[0]) + gt[1]) # append Y

        ptr.append(-np.sin(pt[2]) * (ptc[i][1] - pt[1]) + np.cos(pt[2]) * (ptc[i][0] - pt[0]) + pt[0]) # append X
        ptr.append(np.cos(pt[2]) * (ptc[i][1] - pt[1]) + np.sin(pt[2]) * (ptc[i][0] - pt[0]) + pt[1]) # append Y
    print(f'gtc: {gtc}')
    print(f'gtr: {gtr}\n')
    print(np.array(gtr)[:8].reshape((4, 2)), '\n')
    print(f'ptr: {ptr}\n')
    print(rbox_iou_shapely(gtr, ptr))

def test_detectron2():
    w = 70
    h = 100
    gt = torch.tensor([[100, 100, w, h, 0.2],
                       [150, 100, w, h, 0.2],
                       [150, 100, w, h, 0.2]])
    dt = torch.tensor([[100, 100, w, h, 1.0],
                       [100, 150, w, h, 1.5]])

    print(rbox_iou_d2(dt, gt))

if __name__=='__main__':
    # test_shapely()
    test_detectron2()