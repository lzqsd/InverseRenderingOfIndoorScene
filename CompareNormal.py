import glob
import cv2
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage

gtRoot = '/home/zhl/CVPR20/Resubmission/Dataset/NYU/normals/'
maskRoot = '/home/zhl/CVPR20/Resubmission/Dataset/NYU/masks/'
predRoot = 'NYU_cascade1/results_brdf2_brdf3/'
suffix = '_normal1.npy'

normalNames = glob.glob(osp.join(predRoot, '*' + suffix) )
thetaTotalMean = 0
thetaTotalMedian = 0
cnt = 0
for normalName in normalNames:
    normal = np.load(normalName )
    normal = cv2.resize(normal, (640, 480), cv2.INTER_LINEAR)

    normalGtName = normalName.replace(suffix, '.png').replace(predRoot, gtRoot)
    maskGtName = normalName.replace(suffix, '.png').replace(predRoot, maskRoot )

    mask = cv2.imread(maskGtName )[:, :, ::-1]
    normalGt = cv2.imread(normalGtName )[:, :, ::-1]

    mask = np.min(mask[:, :, :], axis=2)
    mask = (mask == 255 )
    mask = mask.astype(np.float32)[:, :, np.newaxis]
    normalGt = normalGt.astype(np.float32 )

    normalGt = normalGt.astype(np.float32 )
    normalGt = (normalGt - 127.5) / 127.5
    normalNorm = np.sqrt(np.sum(normalGt * normalGt, axis=2 ) )[:, :, np.newaxis]

    normalGt = normalGt / np.sqrt(np.sum( (normalGt * normalGt ), axis=2 )[:, :, np.newaxis] )

    cosTheta =  np.clip(np.sum(normal * normalGt, axis=2), -1, 1)
    theta = np.arccos(cosTheta ) / np.pi * 180

    thetaMean = np.sum(theta * mask[:, :, 0] ) / np.sum(mask[:, :, 0] )
    thetaMedian = np.median(theta[mask[:, :, 0] != 0]  )

    thetaTotalMean += thetaMean
    thetaTotalMedian += thetaMedian
    cnt += 1
    print('Current mean: %.3f Accumulate Mean: %.3f' % (thetaMean, thetaTotalMean / cnt ) )
    print('Current median: %.3f Accumulate Median: %.3f' % (thetaMedian, thetaTotalMedian / cnt ) )

