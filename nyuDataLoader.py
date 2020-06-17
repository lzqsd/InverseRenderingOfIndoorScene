import glob
import numpy as np
import os.path as osp
from PIL import Image
import random
import struct
from torch.utils.data import Dataset
import scipy.ndimage as ndimage
import cv2
from skimage.measure import block_reduce
import json
import scipy.ndimage as ndimage


class ConcatDataset(Dataset ):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets )

    def __len__(self ):
        return max(len(d) for d in self.datasets )



class NYULoader(Dataset ):
    def __init__(self, imRoot, normalRoot, depthRoot, segRoot,
            imHeight = 480, imWidth = 640,
            imWidthMax = 600, imWidthMin = 560,
            phase='TRAIN', rseed = None ):

        self.imRoot = imRoot
        self.imHeight = imHeight
        self.imWidth = imWidth
        self.phase = phase.upper()

        self.imWidthMax = imWidthMax
        self.imWidthMin = imWidthMin


        if phase == 'TRAIN':
            with open('NYUTrain.txt', 'r') as fIn:
                imList = fIn.readlines()
            self.imList = [osp.join(self.imRoot, x.strip() ) for x in imList ]
        elif phase == 'TEST':
            with open('NYUTest.txt', 'r') as fIn:
                imList = fIn.readlines()
            self.imList = [osp.join(self.imRoot, x.strip() ) for x in imList ]


        self.normalList = [x.replace(imRoot, normalRoot) for x in self.imList ]
        self.segList = [x.replace(imRoot, segRoot) for x in self.imList ]
        self.depthList = [x.replace(imRoot, depthRoot).replace('.png', '.tiff') for x in self.imList]

        print('Image Num: %d' % len(self.imList) )

        # Permute the image list
        self.count = len(self.imList )
        self.perm = list(range(self.count ) )

        if rseed is not None:
            random.seed(0)
        random.shuffle(self.perm )

    def __len__(self):
        return len(self.perm )

    def __getitem__(self, ind):

        ind = (ind % len(self.perm) )
        if ind == 0:
            random.shuffle(self.perm )

        if self.phase == 'TRAIN':
            scale = np.random.random();
            imCropWidth = int( np.round( (self.imWidthMax - self.imWidthMin ) * scale + self.imWidthMin ) )
            imCropHeight = int( float(self.imHeight) / float(self.imWidth ) * imCropWidth )
            rs = int(np.round( (480 - imCropHeight) * np.random.random() ) )
            re = rs + imCropHeight
            cs = int(np.round( (640 - imCropWidth) * np.random.random() ) )
            ce = cs + imCropWidth
        elif self.phase == 'TEST':
            imCropWidth = self.imWidth
            imCropHeight = self.imHeight
            rs, re, cs, ce = 0, 480, 0, 640

        segNormal = 0.5 * ( self.loadImage(self.segList[self.perm[ind] ], rs, re, cs, ce) + 1)[0:1, :, :]

        # Read Image
        im = 0.5 * (self.loadImage(self.imList[self.perm[ind] ], rs, re, cs, ce, isGama = True ) + 1)

        # normalize the normal vector so that it will be unit length
        normal = self.loadImage( self.normalList[self.perm[ind] ], rs, re, cs, ce )
        normal = normal / np.sqrt(np.maximum(np.sum(normal * normal, axis=0), 1e-5) )[np.newaxis, :]

        # Read depth
        depth = self.loadDepth(self.depthList[self.perm[ind] ], rs, re, cs, ce )
        if imCropHeight != self.imHeight or imCropWidth != self.imWidth:
            depth = np.squeeze(depth, axis=0)
            depth = cv2.resize(depth, (self.imWidth, self.imHeight), interpolation = cv2.INTER_LINEAR)
            depth = depth[np.newaxis, :, :]
        segDepth = np.logical_and(depth > 1, depth < 10).astype(np.float32 )

        if imCropHeight != self.imHeight or imCropWidth != self.imWidth:
            normal = normal.transpose([1, 2, 0] )
            normal = cv2.resize(normal, (self.imWidth, self.imHeight), interpolation = cv2.INTER_LINEAR)
            normal = normal.transpose([2, 0, 1] )
        normal = normal / np.maximum(np.sqrt(np.sum(normal * normal, axis=0 )[np.newaxis, :, :] ), 1e-5)

        if imCropHeight != self.imHeight or imCropWidth != self.imWidth:
            segNormal = np.squeeze(segNormal, axis=0)
            segNormal = cv2.resize(segNormal, (self.imWidth, self.imHeight), interpolation = cv2.INTER_LINEAR)
            segNormal = segNormal[np.newaxis, :, :]

            im = im.transpose([1, 2, 0] )
            im = cv2.resize(im, (self.imWidth, self.imHeight), interpolation = cv2.INTER_LINEAR )
            im = im.transpose([2, 0, 1] )

        if self.phase == 'TRAIN':
            if np.random.random() > 0.5:
                normal = np.ascontiguousarray(normal[:, :, ::-1] )
                normal[0, :, :] = -normal[0, :, :]
                depth = np.ascontiguousarray(depth[:, :, ::-1] )
                segNormal = np.ascontiguousarray(segNormal[:, :, ::-1] )
                segDepth = np.ascontiguousarray(segDepth[:, :, ::-1] )
                im = np.ascontiguousarray(im[:, :, ::-1] )
            scale = 1 + ( np.random.random(3) * 0.4 - 0.2 )
            scale = scale.reshape([3, 1, 1] )
            im = im * scale


        batchDict = {'normal': normal,
                'depth': depth,
                'segNormal': segNormal,
                'segDepth': segDepth,
                'im': im.astype(np.float32 ),
                'name': self.imList[self.perm[ind] ]
                } 

        return batchDict 

    def loadImage(self, imName, rs, re, cs, ce, isGama = False):
        if not(osp.isfile(imName ) ):
            print(imName )
            assert(False )

        im = cv2.imread(imName)
        if len(im.shape ) == 3:
            im = im[:, :, ::-1]

        im = im[rs:re, cs:ce, :]
        im = np.ascontiguousarray(im.astype(np.float32 ) )
        if isGama:
            im = (im / 255.0) ** 2.2
            im = 2 * im - 1
        else:
            im = (im - 127.5) / 127.5
        if len(im.shape) == 2:
            im = im[:, np.newaxis]
        im = np.transpose(im, [2, 0, 1] )

        return im

    def loadDepth(self, imName, rs, re, cs, ce ):
        if not osp.isfile(imName):
            print(imName )
            assert(False )

        im = cv2.imread(imName, -1)
        im = im[rs:re, cs:ce]
        im = im[np.newaxis, :, :]
        return im
