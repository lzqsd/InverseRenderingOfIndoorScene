from __future__ import print_function
import numpy as np
from PIL import Image
import cv2
import os.path as osp
import torch
from torch.autograd import Variable 
import h5py

def srgb2rgb(srgb ):
    ret = np.zeros_like(srgb )
    idx0 = srgb <= 0.04045
    idx1 = srgb > 0.04045
    ret[idx0] = srgb[idx0] / 12.92
    ret[idx1] = np.power( (srgb[idx1] + 0.055) / 1.055, 2.4 )
    return ret

def writeErrToScreen(errorName, errorArr, epoch, j):
    print( ('[%d/%d] {0}:' % (epoch, j) ).format(errorName), end=' ')
    for n in range(0, len(errorArr) ):
        print('%.6f' % errorArr[n].data.item(), end = ' ')
    print('.')

def writeCoefToScreen(coefName, coef, epoch, j):
    print( ('[%d/%d] {0}:' % (epoch, j) ).format(coefName), end=' ')
    coefNp = coef.cpu().data.numpy()
    for n in range(0, len(coefNp) ):
        print('%.6f' % coefNp[n], end = ' ')
    print('.')

def writeNpErrToScreen(errorName, errorArr, epoch, j):
    print( ('[%d/%d] {0}:' % (epoch, j) ).format(errorName), end=' ')
    for n in range(0, len(errorArr) ):
        print('%.6f' % errorArr[n], end = ' ')
    print('.')


def writeErrToFile(errorName, errorArr, fileOut, epoch, j):
    fileOut.write( ('[%d/%d] {0}:'% (epoch, j) ).format(errorName) )
    for n in range(0, len(errorArr) ):
        fileOut.write('%.6f ' % errorArr[n].data.item() )
    fileOut.write('.\n')

def writeCoefToFile(coefName, coef, fileOut, epoch, j):
    fileOut.write( ('[%d/%d] {0}: ' % (epoch, j) ).format(coefName) )
    coefNp = coef.cpu().data.numpy()
    for n in range(0, len(coefNp) ):
        fileOut.write('%.6f ' % coefNp[n] )
    fileOut.write('.\n')

def writeNpErrToFile(errorName, errorArr, fileOut, epoch, j):
    fileOut.write( ('[%d/%d] {0}:' % (epoch, j) ).format(errorName) )
    for n in range(0, len(errorArr) ):
        fileOut.write('%.6f ' % errorArr[n] )
    fileOut.write('.\n')

def turnErrorIntoNumpy(errorArr):
    errorNp = []
    for n in range(0, len(errorArr) ):
        errorNp.append(errorArr[n].data.item() )
    return np.array(errorNp)[np.newaxis, :]



def writeImageToFile(imgBatch, nameBatch, isGama = False):
    batchSize = imgBatch.size(0)
    for n in range(0, batchSize):
        img = imgBatch[n, :, :, :].data.cpu().numpy()
        img = np.clip(img, 0, 1)
        if isGama:
            img = np.power(img, 1.0/2.2)
        img = (255 *img.transpose([1, 2, 0] ) ).astype(np.uint8)
        if img.shape[2] == 1:
            img = np.concatenate([img, img, img], axis=2)
        img = Image.fromarray(img )
        img.save(nameBatch[n] )


def writeNumpyToFile(imBatch, nameBatch):
    batchSize = imBatch.size(0)
    for n in range(0, batchSize):
        im = imBatch[n, :, :, :].data.cpu().numpy()
        np.save(nameBatch[n], im)

def writeNumpzToFile(imBatch, nameBatch):
    batchSize = imBatch.size(0)
    for n in range(0, batchSize):
        im = imBatch[n, :, :, :].data.cpu().numpy()
        np.savez_compressed(nameBatch[n], data = im)


def writeH5ToFile(imBatch, nameBatch):
    batchSize = imBatch.size(0)
    assert(batchSize == len(nameBatch ) )
    for n in range(0, batchSize):
        im = imBatch[n, :, :, :].data.cpu().numpy()
        hf = h5py.File(nameBatch[n], 'w')
        hf.create_dataset('data', data=im, compression = 'lzf')
        hf.close()


def writeEnvToFile(envmaps, envId, envName, nrows=12, ncols=8, envHeight=8, envWidth=16, gap=1):
    envmap = envmaps[envId, :, :, :, :, :].data.cpu().numpy()
    envmap = np.transpose(envmap, [1, 2, 3, 4, 0] )
    envRow, envCol = envmap.shape[0], envmap.shape[1]

    interY = int(envRow / nrows )
    interX = int(envCol / ncols )

    lnrows = len(np.arange(0, envRow, interY) )
    lncols = len(np.arange(0, envCol, interX) )

    lenvHeight = lnrows * (envHeight + gap) + gap
    lenvWidth = lncols * (envWidth + gap) + gap

    envmapLarge = np.zeros([lenvHeight, lenvWidth, 3], dtype=np.float32) + 1.0
    for r in range(0, envRow, interY ):
        for c in range(0, envCol, interX ):
            rId = int(r / interY )
            cId = int(c / interX )

            rs = rId * (envHeight + gap )
            cs = cId * (envWidth + gap )
            envmapLarge[rs : rs + envHeight, cs : cs + envWidth, :] = envmap[r, c, :, :, :]

    envmapLarge = np.clip(envmapLarge, 0, 1)
    envmapLarge = (255 * (envmapLarge ** (1.0/2.2) ) ).astype(np.uint8 )
    cv2.imwrite(envName, envmapLarge[:, :, ::-1] )

def writeNumpyEnvToFile(envmap, envName, nrows=12, ncols=8, envHeight=8, envWidth=16, gap=1):
    envRow, envCol = envmap.shape[0], envmap.shape[1]

    interY = int(envRow / nrows )
    interX = int(envCol / ncols )

    lnrows = len(np.arange(0, envRow, interY) )
    lncols = len(np.arange(0, envCol, interX) )

    lenvHeight = lnrows * (envHeight + gap) + gap
    lenvWidth = lncols * (envWidth + gap) + gap

    envmapLarge = np.zeros([lenvHeight, lenvWidth, 3], dtype=np.float32) + 1.0
    for r in range(0, envRow, interY ):
        for c in range(0, envCol, interX ):
            rId = int(r / interY )
            cId = int(c / interX )

            rs = rId * (envHeight + gap )
            cs = cId * (envWidth + gap )
            envmapLarge[rs : rs + envHeight, cs : cs + envWidth, :] = envmap[r, c, :, :, :]

    envmapLarge = np.clip(envmapLarge, 0, 1)
    envmapLarge = (255 * envmapLarge ** (1.0/2.2) ).astype(np.uint8 )
    cv2.imwrite(envName, envmapLarge[:, :, ::-1] )

def predToShading(pred, envWidth = 32, envHeight = 16, SGNum = 12 ):

    Az = ( (np.arange(envWidth) + 0.5) / envWidth - 0.5 )* 2 * np.pi
    El = ( (np.arange(envHeight) + 0.5) / envHeight) * np.pi / 2.0
    Az, El = np.meshgrid(Az, El)
    Az = Az[np.newaxis, :, :]
    El = El[np.newaxis, :, :]
    lx = np.sin(El) * np.cos(Az)
    ly = np.sin(El) * np.sin(Az)
    lz = np.cos(El)
    ls = np.concatenate((lx, ly, lz), axis = 0)
    ls = ls[np.newaxis, :, np.newaxis, np.newaxis, :, :]
    envWeight = np.cos(El) * np.sin(El )
    envWeight = envWeight[np.newaxis, np.newaxis, np.newaxis, :, :]

    envRow, envCol = pred.shape[2], pred.shape[3]
    pred = pred.squeeze(0)
    axisOrig = pred[0:3*SGNum, :, :]
    lambOrig = pred[3*SGNum : 4*SGNum, :, :]
    weightOrig = pred[4*SGNum : 7*SGNum, :, :]

    weight = weightOrig.reshape([SGNum, 3, envRow, envCol] ) * 0.999
    weight = np.tan(np.pi / 2.0 * weight )
    weight = weight[:, :, :, :, np.newaxis, np.newaxis ]

    axisDir = axisOrig.reshape([SGNum, 3, envRow, envCol] )
    axisDir = axisDir[:, :, :, :, np.newaxis, np.newaxis]

    lamb = lambOrig.reshape([SGNum, 1, envRow, envCol] ) * 0.999
    lamb = np.tan(np.pi / 2.0 * lamb )
    lamb = lamb[:, :, :, :, np.newaxis, np.newaxis]

    mi = lamb * (np.sum(axisDir * ls, axis=1)[:, np.newaxis, :, :, :, :] - 1)
    envmaps = np.sum(weight * np.exp(mi ), axis=0)

    shading = (envmaps * envWeight ).reshape([3, envRow, envCol, -1] )
    shading = np.sum(shading, axis = 3)
    shading = np.maximum(shading, 0.0)

    return shading
