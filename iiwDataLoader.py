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


class ConcatDataset(Dataset ):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets )

    def __len__(self ):
        return max(len(d) for d in self.datasets )


class IIWLoader(Dataset ):
    def __init__(self, dataRoot, imHeight = 480, imWidth = 640, phase='TRAIN',
            rseed = None, maxNum = 800 ):
        self.dataRoot = dataRoot
        self.imHeight = imHeight
        self.imWidth = imWidth
        self.phase = phase.upper()
        self.maxNum = maxNum

        if phase == 'TRAIN':
            with open('IIWTrain.txt', 'r') as fIn:
                imList = fIn.readlines()
            self.imList = [osp.join(self.dataRoot, x.strip()) for x in imList ]
        elif phase == 'TEST':
            with open('IIWTest.txt', 'r') as fIn:
                imList = fIn.readlines()
            self.imList = [osp.join(self.dataRoot, x.strip()) for x in imList ]

        print('Image Num: %d' % len(self.imList) )
        self.jsonList = [x.replace('.png', '.json') for x in self.imList ]

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

        judgements = json.load(open(self.jsonList[self.perm[ind] ] )  )

        # Read Image
        eqPoint, eqWeight, darkerPoint, darkerWeight, im \
                = self.loadImage(self.imList[self.perm[ind] ], judgements, isGama = True)
        assert(eqPoint.shape[0] == eqWeight.shape[0] )
        assert(darkerPoint.shape[0] == darkerWeight.shape[0] )

        eqNum = eqPoint.shape[0]
        if eqNum < self.maxNum:
            gap = self.maxNum - eqNum
            eqPoint = np.concatenate([eqPoint, np.zeros( (gap, 4), dtype=np.long) ], axis=0 )
            eqWeight = np.concatenate([eqWeight, np.zeros(gap, dtype=np.float32)], axis=0 )
        elif eqNum > self.maxNum:
            index = np.random.permutation(np.arange(eqNum ) )
            eqPoint = eqPoint[index, :]
            eqWeight = eqWeight[index ]

            eqPoint = eqPoint[0:self.maxNum, :]
            eqWeight = eqWeight[0:self.maxNum ]
            eqNum = self.maxNum

        darkerNum = darkerPoint.shape[0]
        if darkerNum < self.maxNum:
            gap = self.maxNum - darkerNum
            darkerPoint = np.concatenate([darkerPoint, np.zeros( (gap, 4), dtype=np.long) ], axis=0 )
            darkerWeight = np.concatenate([darkerWeight, np.zeros(gap, dtype=np.float32)], axis=0 )
        elif darkerNum > self.maxNum:
            index = np.random.permutation(np.arange(darkerNum ) )
            darkerPoint = darkerPoint[index, :]
            darkerWeight = darkerWeight[index ]

            darkerPoint = darkerPoint[0:self.maxNum, :]
            darkerWeight = darkerWeight[0:self.maxNum]
            darkerNum = self.maxNum

        batchDict = {'im': im,
                'eq': {'point' : eqPoint, 'weight' : eqWeight, 'num': eqNum },
                'darker': {'point' : darkerPoint, 'weight' : darkerWeight, 'num' : darkerNum },
                'name': self.imList[self.perm[ind] ]
                }


        return batchDict


    def loadImage(self, imName, judgements, isGama = False):
        if not(osp.isfile(imName ) ):
            print(imName )
            assert(False )

        im = Image.open(imName)
        nw, nh = im.size

        scaleW = float(self.imWidth ) / float(nw )
        scaleH = float(self.imHeight ) / float(nh )
        if scaleW > scaleH:
            newW = self.imWidth
            newH = int( np.ceil(scaleW * nh ) )
            assert(newW >= self.imWidth and newH >= self.imHeight )
            im = im.resize([newW, newH], Image.ANTIALIAS )
            cs, ce = 0, self.imWidth
            gap = newH - self.imHeight
            rs = np.random.randint(gap + 1)
            re = rs + self.imHeight
        else:
            newH = self.imHeight
            newW = int(np.ceil(scaleH * nw ) )
            assert(newW >= self.imWidth and newH >= self.imHeight )
            im = im.resize([newW, newH], Image.ANTIALIAS )
            rs, re = 0, self.imHeight
            gap = newW - self.imWidth
            cs = np.random.randint(gap + 1)
            ce = cs + self.imWidth

        im = np.asarray(im, dtype=np.float32)
        im = im / 255.0

        points = judgements['intrinsic_points']
        comparisons = judgements['intrinsic_comparisons']
        id_to_points = {p['id']: p for p in points}

        eqPoint, eqWeight = [0, 0, 0, 0], [0]
        darkerPoint, darkerWeight = [0, 0, 0, 0], [0]
        for c in comparisons:
            darker = c['darker']
            if darker not in ('1', '2', 'E'):
                continue

            # "darker_score" is "w_i" in our paper
            weight = c['darker_score']
            if weight <= 0.0 or weight is None:
                continue

            point1 = id_to_points[c['point1']]
            point2 = id_to_points[c['point2']]
            if not point1['opaque'] or not point2['opaque']:
                continue

            r1, c1 = int(point1['y'] * newH ), int(point1['x'] * newW )
            r2, c2 = int(point2['y'] * newH ), int(point2['x'] * newW )

            pr1 = float(r1 - rs) / float(self.imHeight -1 )
            pc1 = float(c1 - cs ) / float(self.imWidth - 1 )
            pr2 = float(r2 - rs ) / float(self.imHeight - 1 )
            pc2 = float(c2 - cs ) / float(self.imWidth - 1 )

            if not pr1 >= 0.0 or not pr1 <= 1.0:
                continue
            assert(pr1 >= 0.0 and pr1 <= 1.0)
            if pc1 < 0.0 or pc1 > 1.0:
                continue
            assert(pc1 >= 0.0 and pc1 <= 1.0)
            if not pr2 >= 0.0 or not pr2 <= 1.0:
                continue
            assert(pr2 >= 0.0 and pr2 <= 1.0)
            if pc2 < 0.0 or pc2 > 1.0:
                continue
            assert(pc2 >= 0.0 and pc2 <= 1.0)

            prId1 = int(pr1 * (self.imHeight - 1) )
            pcId1 = int(pc1 * (self.imWidth - 1) )
            prId2 = int(pr2 * (self.imHeight - 1) )
            pcId2 = int(pc2 * (self.imWidth - 1) )


            # the second point should be darker than the first point
            if darker == 'E':
                eqPoint = eqPoint + [prId1, pcId1, prId2, pcId2 ]
                eqWeight.append(weight )
            elif darker == '1':
                darkerPoint = darkerPoint + [prId2, pcId2, prId1, pcId1 ]
                darkerWeight.append(weight )
            elif darker == '2':
                darkerPoint = darkerPoint + [prId1, pcId1, prId2, pcId2 ]
                darkerWeight.append(weight )

        eqWeight = np.asarray(eqWeight, dtype=np.float32 )
        eqPoint = np.asarray(eqPoint, dtype=np.long )
        eqPoint = eqPoint.reshape([-1, 4] )
        darkerWeight = np.asarray(darkerWeight, dtype=np.float32 )
        darkerPoint = np.asarray(darkerPoint, dtype=np.float32 )
        darkerPoint = darkerPoint.reshape([-1, 4] )

        if isGama:
            im = im ** (2.2)

        im = im[rs:re, cs:ce, :]
        assert(im.shape[0] == self.imHeight and im.shape[1] == self.imWidth )

        if len(im.shape) == 2:
            im = im[:, np.newaxis]
        im = np.transpose(im, [2, 0, 1] )
        im = im / im.max()

        return eqPoint, eqWeight, darkerPoint, darkerWeight, im

    def loadNumpy(self, imName):
        im = np.load(imName )
        return im

    def loadNumpz(self, imName ):
        im = np.load(imName )
        return im['data']


    def loadEnvmapPred(self, envName ):
        if not osp.isfile(envName ):
            print('Wrong: %s does not exist' % envName )
            assert(False )
        envmap = np.load(envName )
        return envmap['data']

