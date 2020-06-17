import glob
import cv2
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage

gtRoot = '/home/zhl/CVPR20/Resubmission/Dataset/NYU/depths/'
predRoot = 'NYU_cascade1/results_brdf2_brdf3/'
suffix = '_depthBS1.npy'

depthNames = glob.glob(osp.join(predRoot, '*' + suffix) )
errorAccu = 0
cnt = 0
for depthName in depthNames:
    depth = np.load(depthName )
    depthGtName = depthName.replace(suffix, '.tiff').replace(predRoot, gtRoot)

    depthGt = cv2.imread(depthGtName, -1)
    mask = np.logical_and(depthGt > 1, depthGt < 10).astype(np.float32)

    depth = cv2.resize(depth, (depthGt.shape[1], depthGt.shape[0]),
            interpolation=cv2.INTER_LINEAR )

    depth = np.log(depth + 1e-20)
    depthGt = np.log(depthGt + 1e-20)

    depth = depth - np.mean(depth);
    depthGt = depthGt - np.mean(depthGt )

    error = np.sum(np.power(depth - depthGt, 2) * mask ) / np.sum(mask )
    errorAccu += np.sqrt(error)

    cnt += 1
    print('Current mean: %.3f Accumulate Mean: %.3f' % (np.sqrt(error), errorAccu / cnt ) )

