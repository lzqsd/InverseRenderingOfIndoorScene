import torch
import numpy as np
from torch.autograd import Variable
import argparse
import random
import os
import os.path as osp
import models
import torchvision.utils as vutils
import utils
import dataLoader
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import wrapperBRDFLight as wcg
import os.path as osp

parser = argparse.ArgumentParser()
# The locationi of training set
parser.add_argument('--dataRoot', default=None, help='path to input images')
parser.add_argument('--experimentBRDF', default=None, help='path to the model for BRDF prediction')
parser.add_argument('--experimentLight', default=None, help='path to the model for light prediction')
parser.add_argument('--mode', default='train', help='run training set or testing set' )
# The basic training setting
parser.add_argument('--nepochBRDF', type=int, default=14, help='the number of epochs for BRDF prediction')
parser.add_argument('--nepochLight', type=int, default=10, help='the number of epochs for light prediction')
parser.add_argument('--batchSize', type=int, default=6, help='input batch size')

parser.add_argument('--imHeight', type=int, default=240, help='the height / width of the input image to network')
parser.add_argument('--imWidth', type=int, default=320, help='the height / width of the input image to network')

parser.add_argument('--envRow', type=int, default=120, help='the number of samples of envmaps in y direction')
parser.add_argument('--envCol', type=int, default=160, help='the number of samples of envmaps in x direction')
parser.add_argument('--envHeight', type=int, default=8, help='the size of envmaps in y direction')
parser.add_argument('--envWidth', type=int, default=16, help='the size of envmaps in x direction')
parser.add_argument('--SGNum', type=int, default=12, help='the number of spherical Gaussian lobe' )
parser.add_argument('--offset', type=float, default=1.0, help='the offset for training lighting prediction')

parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--deviceIds', type=int, nargs='+', default=[0], help='the gpus used for training network')
# Cascae Level
parser.add_argument('--cascadeLevel', type=int, default=0, help='the casacade level')

# The detail network setting
opt = parser.parse_args()
print(opt)

opt.gpuId = opt.deviceIds[0]

curDir = '/'.join(osp.abspath(__file__).split('/')[0:-1] )
if opt.experimentLight is None:
    opt.experimentLight = 'check_cascadeLight%d_sg%d_offset%.1f' % \
            (opt.cascadeLevel, opt.SGNum, opt.offset )

if opt.experimentBRDF is None:
    opt.experimentBRDF = 'check_cascade%d_w%d_h%d' \
            % (opt.cascadeLevel, opt.imWidth, opt.imHeight )

opt.experimentLight = osp.join(curDir, opt.experimentLight )
opt.experimentBRDF = osp.join(curDir, opt.experimentBRDF )

opt.seed = 45
print("Random Seed: ", opt.seed )
random.seed(opt.seed )
torch.manual_seed(opt.seed )

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

####################################
# Initial Network
encoder = models.encoder0(cascadeLevel = opt.cascadeLevel )
albedoDecoder = models.decoder0(mode=0 )
normalDecoder = models.decoder0(mode=1 )
roughDecoder = models.decoder0(mode=2 )
depthDecoder = models.decoder0(mode=4 )

lightEncoder = models.encoderLight(cascadeLevel = opt.cascadeLevel,
        SGNum = opt.SGNum )
axisDecoder = models.decoderLight(mode=0, SGNum = opt.SGNum )
lambDecoder = models.decoderLight(mode = 1, SGNum = opt.SGNum )
weightDecoder = models.decoderLight(mode = 2, SGNum = opt.SGNum )

renderLayer = models.renderingLayer(isCuda = opt.cuda,
        imWidth=opt.envCol, imHeight=opt.envRow,
        envWidth = opt.envWidth, envHeight = opt.envHeight)

output2env = models.output2env(isCuda = opt.cuda,
        envWidth = opt.envWidth, envHeight = opt.envHeight, SGNum = opt.SGNum )
####################################################################


#########################################
encoder.load_state_dict(
        torch.load('{0}/encoder{1}_{2}.pth'.format(opt.experimentBRDF,
            opt.cascadeLevel, opt.nepochBRDF-1) ).state_dict() )
albedoDecoder.load_state_dict(
        torch.load('{0}/albedo{1}_{2}.pth'.format(opt.experimentBRDF,
            opt.cascadeLevel, opt.nepochBRDF-1) ).state_dict() )
normalDecoder.load_state_dict(
        torch.load('{0}/normal{1}_{2}.pth'.format(opt.experimentBRDF,
            opt.cascadeLevel, opt.nepochBRDF-1) ).state_dict() )
roughDecoder.load_state_dict(
        torch.load('{0}/rough{1}_{2}.pth'.format(opt.experimentBRDF,
            opt.cascadeLevel, opt.nepochBRDF-1) ).state_dict() )
depthDecoder.load_state_dict(
        torch.load('{0}/depth{1}_{2}.pth'.format(opt.experimentBRDF,
            opt.cascadeLevel, opt.nepochBRDF-1) ).state_dict() )

lightEncoder.load_state_dict(
        torch.load('{0}/lightEncoder{1}_{2}.pth'.format(opt.experimentLight,
            opt.cascadeLevel, opt.nepochLight-1 ) ).state_dict() )
axisDecoder.load_state_dict(
        torch.load('{0}/axisDecoder{1}_{2}.pth'.format(opt.experimentLight,
            opt.cascadeLevel, opt.nepochLight-1 ) ).state_dict() )
lambDecoder.load_state_dict(
        torch.load('{0}/lambDecoder{1}_{2}.pth'.format(opt.experimentLight,
            opt.cascadeLevel, opt.nepochLight-1 ) ).state_dict() )
weightDecoder.load_state_dict(
        torch.load('{0}/weightDecoder{1}_{2}.pth'.format(opt.experimentLight,
            opt.cascadeLevel, opt.nepochLight-1 ) ).state_dict() )

for param in encoder.parameters():
    param.requires_grad = False
for param in albedoDecoder.parameters():
    param.requires_grad = False
for param in normalDecoder.parameters():
    param.requires_grad = False
for param in roughDecoder.parameters():
    param.requires_grad = False
for param in depthDecoder.parameters():
    param.requires_grad = False

for param in lightEncoder.parameters():
    param.requires_grad = False
for param in axisDecoder.parameters():
    param.requires_grad = False
for param in lambDecoder.parameters():
    param.requires_grad = False
for param in weightDecoder.parameters():
    param.requires_grad = False

#########################################
encoder = nn.DataParallel(encoder, device_ids = opt.deviceIds )
albedoDecoder = nn.DataParallel(albedoDecoder, device_ids = opt.deviceIds )
normalDecoder = nn.DataParallel(normalDecoder, device_ids = opt.deviceIds )
roughDecoder = nn.DataParallel(roughDecoder, device_ids = opt.deviceIds )
depthDecoder = nn.DataParallel(depthDecoder, device_ids = opt.deviceIds )

lightEncoder = nn.DataParallel(lightEncoder, device_ids = opt.deviceIds )
axisDecoder = nn.DataParallel(axisDecoder, device_ids = opt.deviceIds )
lambDecoder = nn.DataParallel(lambDecoder, device_ids = opt.deviceIds )
weightDecoder = nn.DataParallel(weightDecoder, device_ids = opt.deviceIds )

##############  ######################
# Send things into GPU
if opt.cuda:
    encoder = encoder.cuda(opt.gpuId )
    albedoDecoder = albedoDecoder.cuda(opt.gpuId )
    normalDecoder = normalDecoder.cuda(opt.gpuId )
    roughDecoder = roughDecoder.cuda(opt.gpuId )
    depthDecoder = depthDecoder.cuda(opt.gpuId )

    lightEncoder = lightEncoder.cuda(opt.gpuId )
    axisDecoder = axisDecoder.cuda(opt.gpuId )
    lambDecoder = lambDecoder.cuda(opt.gpuId )
    weightDecoder = weightDecoder.cuda(opt.gpuId )
####################################


####################################

####################################
brdfDataset = dataLoader.BatchLoader( opt.dataRoot, phase = opt.mode,
        imWidth = opt.imWidth, imHeight = opt.imHeight, isLight = True,
        cascadeLevel = opt.cascadeLevel )
brdfLoader = DataLoader(brdfDataset, batch_size = opt.batchSize, num_workers =
        16, shuffle = False )

j = 0
# BRDFLost
albedoErrsNpList = np.ones( [1, 1], dtype = np.float32 )
normalErrsNpList = np.ones( [1, 1], dtype = np.float32 )
roughErrsNpList= np.ones( [1, 1], dtype = np.float32 )
depthErrsNpList = np.ones( [1, 1], dtype = np.float32 )

reconstErrsNpList = np.ones( [1, 1], dtype = np.float32 )
renderErrsNpList = np.ones( [1, 1], dtype = np.float32 )
epoch = opt.nepochBRDF
for i, dataBatch in enumerate(brdfLoader):
    j += 1
    #####################################################################################################################
    ############################################# Test with CGBRDF dataset ##############################################
    #####################################################################################################################
    albedoPair, normalPair, roughPair, depthPair,  \
    envmapsPair, renderPair, lightingOutput\
    = wcg.wrapperBRDFLight(dataBatch, opt, encoder, \
    albedoDecoder, normalDecoder, roughDecoder, depthDecoder, \
    lightEncoder, axisDecoder, lambDecoder, weightDecoder, \
    output2env, renderLayer, isLightOut = True, offset = opt.offset )

    albedoPred, albedoErr = albedoPair[0], albedoPair[1]
    normalPred, normalErr = normalPair[0], normalPair[1]
    roughPred, roughErr = roughPair[0], roughPair[1]
    depthPred, depthErr = depthPair[0], depthPair[1]
    envmapsPredScaledImage, reconstErr = envmapsPair[0], envmapsPair[1]
    renderedImPred, renderErr = renderPair[0], renderPair[1]
    envmapsPred, diffusePred, specularPred = lightingOutput[0], lightingOutput[1], lightingOutput[2]

    # Output training error
    utils.writeErrToScreen('albedo', [albedoErr], epoch, j)
    utils.writeErrToScreen('normal', [normalErr], epoch, j)
    utils.writeErrToScreen('rough', [roughErr], epoch, j)
    utils.writeErrToScreen('depth', [depthErr], epoch, j)

    utils.writeErrToScreen('reconstErrors', [reconstErr], epoch, j)
    utils.writeErrToScreen('renderErrors', [renderErr], epoch, j)

    albedoErrsNpList = np.concatenate( [albedoErrsNpList, utils.turnErrorIntoNumpy( [albedoErr] )], axis=0)
    normalErrsNpList = np.concatenate( [normalErrsNpList, utils.turnErrorIntoNumpy( [normalErr] )], axis=0)
    roughErrsNpList = np.concatenate( [roughErrsNpList, utils.turnErrorIntoNumpy( [roughErr] )], axis=0)
    depthErrsNpList = np.concatenate( [depthErrsNpList, utils.turnErrorIntoNumpy( [depthErr] )], axis=0)

    reconstErrsNpList = np.concatenate( [reconstErrsNpList, utils.turnErrorIntoNumpy( [reconstErr] )], axis=0 )
    renderErrsNpList = np.concatenate( [renderErrsNpList, utils.turnErrorIntoNumpy( [renderErr] )], axis=0 )

    utils.writeNpErrToScreen('albedoAccu', np.mean(albedoErrsNpList[1:, :], axis=0), epoch, j )
    utils.writeNpErrToScreen('normalAccu', np.mean(normalErrsNpList[1:, :], axis=0), epoch, j )
    utils.writeNpErrToScreen('roughAccu', np.mean(roughErrsNpList[1:, :], axis=0), epoch, j )
    utils.writeNpErrToScreen('depthAccu', np.mean(depthErrsNpList[1:, :], axis=0), epoch, j )

    utils.writeNpErrToScreen('reconstAccu', np.mean(reconstErrsNpList[1:, :], axis=0), epoch, j )
    utils.writeNpErrToScreen('renderAccu', np.mean(renderErrsNpList[1:, :], axis=0), epoch, j )

    imNameBatch = dataBatch['name']
    print(imNameBatch )
    albedoNameBatch = [x.replace('im_', 'imbaseColor_').replace('.hdr', '_%d.h5' % opt.cascadeLevel)
            for x in imNameBatch]
    normalNameBatch = [x.replace('im_', 'imnormal_').replace('.hdr', '_%d.h5' % opt.cascadeLevel)
            for x in imNameBatch]
    roughNameBatch = [x.replace('im_', 'imroughness_').replace('.hdr', '_%d.h5' % opt.cascadeLevel)
            for x in imNameBatch]
    depthNameBatch = [x.replace('im_', 'imdepth_').replace('.hdr', '_%d.h5' % opt.cascadeLevel)
            for x in imNameBatch]

    envPreNameBatch = [x.replace('im_', 'imenv_').replace('.hdr', '_%d.h5' % opt.cascadeLevel)
            for x in imNameBatch ]
    diffusePreNameBatch = [x.replace('im_', 'imdiffuse_').replace('.hdr', '_%d.h5' % opt.cascadeLevel)
            for x in imNameBatch ]
    specularPreNameBatch = [x.replace('im_', 'imspecular_').replace('.hdr', '_%d.h5' % opt.cascadeLevel)
            for x in imNameBatch ]

    # Save the predicted results
    bn, ch, nrow, ncol = albedoPred.size()
    albedoPred = albedoPred.view(bn, -1)
    albedoPred = albedoPred / torch.clamp(torch.mean(albedoPred, dim=1), min=1e-10).unsqueeze(1) / 3.0
    albedoPred = albedoPred.view(bn, ch, nrow, ncol)
    for n in range(0, albedoPred.size(0 ) ):
        if not osp.isfile(albedoNameBatch[n] ):
            print(albedoNameBatch[n] )
            utils.writeH5ToFile(albedoPred[n:n+1, :], albedoNameBatch[n:n+1] )

    for n in range(0, normalPred.size(0 ) ):
        if not osp.isfile(normalNameBatch[n]):
            print(normalNameBatch[n] )
            utils.writeH5ToFile(normalPred[n:n+1, :], normalNameBatch[n:n+1] )

    for n in range(0, roughPred.size(0) ):
        if not osp.isfile(roughNameBatch[n] ):
            print(roughNameBatch[n] )
            utils.writeH5ToFile(roughPred[n:n+1, :], roughNameBatch[n:n+1] )

    bn, ch, nrow, ncol = depthPred.size()
    depthPred = depthPred.view(bn, -1)
    depthPred = depthPred / torch.clamp(torch.mean(depthPred, dim=1), min=1e-10).unsqueeze(1) / 3.0
    depthPred = depthPred.view(bn, ch, nrow, ncol)
    for n in range(0, depthPred.size(0) ):
        if not osp.isfile(depthNameBatch[n] ):
            print(depthNameBatch[n] )
            utils.writeH5ToFile(depthPred[n:n+1, :], depthNameBatch[n:n+1] )

    for n in range(0, diffusePred.size(0) ):
        if not osp.isfile(diffusePreNameBatch[n] ):
            print(diffusePreNameBatch[n] )
            utils.writeH5ToFile(diffusePred[n:n+1, :],
                    diffusePreNameBatch[n:n+1] )

    for n in range(0, specularPred.size(0) ):
        if not osp.isfile(specularPreNameBatch[n] ):
            print(specularPreNameBatch[n] )
            utils.writeH5ToFile(specularPred[n:n+1, :],
                    specularPreNameBatch[n:n+1] )

    envmapsInd = dataBatch['envmapsInd']
    for n in range(0, envmapsPred.size(0 ) ):
        envmapPred = envmapsPred[n:n+1, :]
        envPreName = envPreNameBatch[n:n+1]
        if envmapsInd[n] == 1:
            if not osp.isfile(envPreName[0] ):
                print(envPreName[0] )
                utils.writeH5ToFile(envmapPred, envPreName )
