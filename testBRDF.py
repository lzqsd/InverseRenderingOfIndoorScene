import torch
import numpy as np
from torch.autograd import Variable
import argparse
import random
import os
import models
import utils
import dataLoader
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.utils as vutils

parser = argparse.ArgumentParser()
# The locationi of testing set
parser.add_argument('--dataRoot', default=None, help='path to real image distorted by water')
parser.add_argument('--experiment', default=None, help='the path to store samples and models')
parser.add_argument('--testRoot', default=None, help='the path to save the testing errors')
# The basic training setting
parser.add_argument('--nepoch0', type=int, default=14, help='the number of epochs for training')
parser.add_argument('--nepoch1', type=int, default=7, help='the number of epochs for training')

parser.add_argument('--batchSize0', type=int, default=16, help='input batch size')
parser.add_argument('--batchSize1', type=int, default=16, help='input batch size')

parser.add_argument('--imHeight0', type=int, default=240, help='the height / width of the input image to network')
parser.add_argument('--imWidth0', type=int, default=320, help='the height / width of the input image to network')
parser.add_argument('--imHeight1', type=int, default=240, help='the height / width of the input image to network')
parser.add_argument('--imWidth1', type=int, default=320, help='the height / width of the input image to network')

parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--deviceIds', type=int, nargs='+', default=[0], help='the gpus used for testing network')
# Cascae Level
parser.add_argument('--cascadeLevel', type=int, default=0, help='the casacade level')
opt = parser.parse_args()
print(opt)

opt.gpuId = opt.deviceIds[0]

if opt.cascadeLevel == 0:
    opt.nepoch = opt.nepoch0
    opt.batchSize = opt.batchSize0
    opt.imHeight, opt.imWidth = opt.imHeight0, opt.imWidth0
elif opt.cascadeLevel == 1:
    opt.nepoch = opt.nepoch1
    opt.batchSize = opt.batchSize1
    opt.imHeight, opt.imWidth = opt.imHeight1, opt.imWidth1

if opt.experiment is None:
    opt.experiment = 'check_cascade%d_w%d_h%d' % (opt.cascadeLevel,
            opt.imWidth, opt.imHeight )

if opt.testRoot is None:
    opt.testRoot = opt.experiment.replace('check', 'test')
os.system('mkdir {0}'.format(opt.testRoot) )
os.system('cp *.py %s' % opt.testRoot )

opt.seed = 0
print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# Initial Network
encoder = models.encoder0(cascadeLevel = opt.cascadeLevel)
albedoDecoder = models.decoder0(mode=0 )
normalDecoder = models.decoder0(mode=1 )
roughDecoder = models.decoder0(mode=2 )
depthDecoder = models.decoder0(mode=4 )
####################################################################


#########################################
encoder.load_state_dict(
        torch.load('{0}/encoder{1}_{2}.pth'.format(opt.experiment, opt.cascadeLevel, opt.nepoch-1 ) ).state_dict() )
albedoDecoder.load_state_dict(
        torch.load('{0}/albedo{1}_{2}.pth'.format(opt.experiment, opt.cascadeLevel, opt.nepoch-1 ) ).state_dict() )
normalDecoder.load_state_dict(
        torch.load('{0}/normal{1}_{2}.pth'.format(opt.experiment, opt.cascadeLevel, opt.nepoch-1 ) ).state_dict() )
roughDecoder.load_state_dict(
        torch.load('{0}/rough{1}_{2}.pth'.format(opt.experiment, opt.cascadeLevel, opt.nepoch-1 ) ).state_dict() )
depthDecoder.load_state_dict(
        torch.load('{0}/depth{1}_{2}.pth'.format(opt.experiment, opt.cascadeLevel, opt.nepoch-1) ).state_dict() )
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

#########################################
encoder = nn.DataParallel(encoder, device_ids = opt.deviceIds )
albedoDecoder = nn.DataParallel(albedoDecoder, device_ids = opt.deviceIds )
normalDecoder = nn.DataParallel(normalDecoder, device_ids = opt.deviceIds )
roughDecoder = nn.DataParallel(roughDecoder, device_ids = opt.deviceIds )
depthDecoder = nn.DataParallel(depthDecoder, device_ids = opt.deviceIds )

##############  ######################
# Send things into GPU
if opt.cuda:
    encoder = encoder.cuda(opt.gpuId)
    albedoDecoder = albedoDecoder.cuda(opt.gpuId)
    normalDecoder = normalDecoder.cuda(opt.gpuId)
    roughDecoder = roughDecoder.cuda(opt.gpuId)
    depthDecoder = depthDecoder.cuda(opt.gpuId)
####################################


####################################
brdfDataset = dataLoader.BatchLoader( opt.dataRoot, imWidth = opt.imWidth, imHeight = opt.imHeight, rseed = opt.seed,
        cascadeLevel = opt.cascadeLevel, phase = 'TEST')
brdfLoader = DataLoader(brdfDataset, batch_size = opt.batchSize, num_workers = 4, shuffle = False)

j = 0
albedoErrsNpList = np.ones( [1, 1], dtype = np.float32 )
normalErrsNpList = np.ones( [1, 1], dtype = np.float32 )
roughErrsNpList= np.ones( [1, 1], dtype = np.float32 )
depthErrsNpList = np.ones( [1, 1], dtype = np.float32 )

epoch = opt.nepoch
testingLog = open('{0}/testingLog_{1}.txt'.format(opt.testRoot, epoch), 'w')
for i, dataBatch in enumerate(brdfLoader):
    j += 1
    # Load data from cpu to gpu
    albedo_cpu = dataBatch['albedo']
    albedoBatch = Variable(albedo_cpu ).cuda()

    normal_cpu = dataBatch['normal']
    normalBatch = Variable(normal_cpu ).cuda()

    rough_cpu = dataBatch['rough']
    roughBatch = Variable(rough_cpu ).cuda()

    depth_cpu = dataBatch['depth']
    depthBatch = Variable(depth_cpu ).cuda()

    segArea_cpu = dataBatch['segArea']
    segEnv_cpu = dataBatch['segEnv']
    segObj_cpu = dataBatch['segObj']
    seg_cpu = torch.cat([segArea_cpu, segEnv_cpu, segObj_cpu], dim=1 )
    segBatch = Variable(seg_cpu ).cuda()

    segBRDFBatch = segBatch[:, 2:3, :, :]
    segAllBatch = segBatch[:, 0:1, :, :]  + segBatch[:, 2:3, :, :]

    # Load the image from cpu to gpu
    im_cpu = (dataBatch['im'] )
    imBatch = Variable(im_cpu ).cuda()

    if opt.cascadeLevel > 0:
        albedoPre_cpu = dataBatch['albedoPre']
        albedoPreBatch = Variable(albedoPre_cpu ).cuda()

        normalPre_cpu = dataBatch['normalPre']
        normalPreBatch = Variable(normalPre_cpu ).cuda()

        roughPre_cpu = dataBatch['roughPre']
        roughPreBatch = Variable(roughPre_cpu ).cuda()

        depthPre_cpu = dataBatch['depthPre']
        depthPreBatch = Variable(depthPre_cpu ).cuda()

        diffusePre_cpu = dataBatch['diffusePre']
        diffusePreBatch = Variable(diffusePre_cpu ).cuda()

        specularPre_cpu = dataBatch['specularPre']
        specularPreBatch = Variable(specularPre_cpu ).cuda()

        if albedoPreBatch.size(2) < opt.imHeight or albedoPreBatch.size(3) < opt.imWidth:
            albedoPreBatch = F.interpolate(albedoPreBatch, [opt.imHeight, opt.imWidth ], mode='bilinear')
        if normalPreBatch.size(2) < opt.imHeight or normalPreBatch.size(3) < opt.imWidth:
            normalPreBatch = F.interpolate(normalPreBatch, [opt.imHeight, opt.imWidth ], mode='bilinear')
        if roughPreBatch.size(2) < opt.imHeight or roughPreBatch.size(3) < opt.imWidth:
            roughPreBatch = F.interpolate(roughPreBatch, [opt.imHeight, opt.imWidth ], mode='bilinear')
        if depthPreBatch.size(2) < opt.imHeight or depthPreBatch.size(3) < opt.imWidth:
            depthPreBatch = F.interpolate(depthPreBatch, [opt.imHeight, opt.imWidth ], mode='bilinear')

        # Regress the diffusePred and specular Pred
        envRow, envCol = diffusePreBatch.size(2), diffusePreBatch.size(3)
        imBatchSmall = F.adaptive_avg_pool2d(imBatch, (envRow, envCol) )
        diffusePreBatch, specularPreBatch = models.LSregressDiffSpec(
                diffusePreBatch, specularPreBatch,
                imBatchSmall, diffusePreBatch, specularPreBatch )

        if diffusePreBatch.size(2) < opt.imHeight or diffusePreBatch.size(3) < opt.imWidth:
            diffusePreBatch = F.interpolate(diffusePreBatch, [opt.imHeight, opt.imWidth ], mode='bilinear')
        if specularPreBatch.size(2) < opt.imHeight or specularPreBatch.size(3) < opt.imWidth:
            specularPreBatch = F.interpolate(specularPreBatch, [opt.imHeight, opt.imWidth ], mode='bilinear')


    ########################################################
    # Build the cascade network architecture #
    albedoPreds = []
    normalPreds = []
    roughPreds = []
    depthPreds = []

    # Initial Prediction
    if opt.cascadeLevel == 0:
        inputBatch = imBatch
    elif opt.cascadeLevel > 0:
        inputBatch = torch.cat([imBatch, albedoPreBatch,
            normalPreBatch, roughPreBatch, depthPreBatch,
            diffusePreBatch, specularPreBatch ], dim=1 )

    x1, x2, x3, x4, x5, x6 = encoder(inputBatch )
    albedoPred = 0.5 * (albedoDecoder(imBatch, x1, x2, x3, x4, x5, x6) + 1)
    normalPred = normalDecoder(imBatch, x1, x2, x3, x4, x5, x6)
    roughPred = roughDecoder(imBatch, x1, x2, x3, x4, x5, x6)
    depthPred = 0.5 * (depthDecoder(imBatch, x1, x2, x3, x4, x5, x6) + 1)

    albedoPred = models.LSregress( albedoPred * segBRDFBatch.expand_as(albedoPred),
            albedoBatch * segBRDFBatch.expand_as(albedoPred), albedoPred )
    albedoPred = torch.clamp(albedoPred, 0, 1 )

    depthPred = models.LSregress(depthPred *  segAllBatch.expand_as(depthPred),
            depthBatch * segAllBatch.expand_as(depthPred), depthPred)

    albedoPreds.append(albedoPred)
    normalPreds.append(normalPred)
    roughPreds.append(roughPred)
    depthPreds.append(depthPred )


    ########################################################

    # Compute the error
    albedoErrs = []
    normalErrs = []
    roughErrs = []
    depthErrs = []

    pixelObjNum = (torch.sum(segBRDFBatch ).cpu().data).item()
    pixelAllNum = (torch.sum(segAllBatch ).cpu().data).item()
    for m in range(0, len(albedoPreds) ):
        albedoErrs.append( torch.sum( (albedoPreds[m] - albedoBatch)
                * (albedoPreds[m] - albedoBatch) * segBRDFBatch.expand_as(albedoBatch) ) / pixelObjNum / 3.0 )
    for m in range(0, len(normalPreds) ):
        normalErrs.append( torch.sum( (normalPreds[m] - normalBatch)
                * (normalPreds[m] - normalBatch) * segAllBatch.expand_as(normalBatch) ) / pixelAllNum / 3.0)
    for m in range(0, len(roughPreds) ):
        roughErrs.append( torch.sum( (roughPreds[m] - roughBatch)
                * (roughPreds[m] - roughBatch) * segBRDFBatch ) / pixelObjNum / 4.0 )
    for n in range(0, len(depthPreds ) ):
        depthErrs.append( torch.sum( (torch.log(depthPreds[n] + 1e-3) - torch.log(depthBatch + 1e-3 ) )
            * ( torch.log(depthPreds[n] + 1e-3 ) - torch.log(depthBatch + 1e-3 ) ) * segAllBatch.expand_as(depthBatch ) ) / pixelAllNum )


    # Output testing error
    utils.writeErrToScreen('albedo', albedoErrs, epoch, j)
    utils.writeErrToScreen('normal', normalErrs, epoch, j)
    utils.writeErrToScreen('rough', roughErrs, epoch, j)
    utils.writeErrToScreen('depth', depthErrs, epoch, j)

    utils.writeErrToFile('albedo', albedoErrs, testingLog, epoch, j)
    utils.writeErrToFile('normal', normalErrs, testingLog, epoch, j)
    utils.writeErrToFile('rough', roughErrs, testingLog, epoch, j)
    utils.writeErrToFile('depth', depthErrs, testingLog, epoch, j)

    albedoErrsNpList = np.concatenate( [albedoErrsNpList, utils.turnErrorIntoNumpy(albedoErrs)], axis=0 )
    normalErrsNpList = np.concatenate( [normalErrsNpList, utils.turnErrorIntoNumpy(normalErrs)], axis=0 )
    roughErrsNpList = np.concatenate( [roughErrsNpList, utils.turnErrorIntoNumpy(roughErrs)], axis=0 )
    depthErrsNpList = np.concatenate( [depthErrsNpList, utils.turnErrorIntoNumpy(depthErrs)], axis=0 )

    utils.writeNpErrToScreen('albedoAccu', np.mean(albedoErrsNpList[1:j+1, :], axis=0), epoch, j )
    utils.writeNpErrToScreen('normalAccu', np.mean(normalErrsNpList[1:j+1, :], axis=0), epoch, j )
    utils.writeNpErrToScreen('roughAccu', np.mean(roughErrsNpList[1:j+1, :], axis=0), epoch, j )
    utils.writeNpErrToScreen('depthAccu', np.mean(depthErrsNpList[1:j+1, :], axis=0), epoch, j )

    utils.writeNpErrToFile('albedoAccu', np.mean(albedoErrsNpList[1:j+1, :], axis=0), testingLog, epoch, j )
    utils.writeNpErrToFile('normalAccu', np.mean(normalErrsNpList[1:j+1, :], axis=0), testingLog, epoch, j )
    utils.writeNpErrToFile('roughAccu', np.mean(roughErrsNpList[1:j+1, :], axis=0), testingLog, epoch, j )
    utils.writeNpErrToFile('depthAccu', np.mean(depthErrsNpList[1:j+1, :], axis=0), testingLog, epoch, j )

    if j == 1 or j% 2000 == 0:
        # Save the ground truth and the input
        vutils.save_image(( (albedoBatch ) ** (1.0/2.2) ).data,
                '{0}/{1}_albedoGt.png'.format(opt.testRoot, j) )
        vutils.save_image( (0.5*(normalBatch + 1) ).data,
                '{0}/{1}_normalGt.png'.format(opt.testRoot, j) )
        vutils.save_image( (0.5*(roughBatch + 1) ).data,
                '{0}/{1}_roughGt.png'.format(opt.testRoot, j) )
        vutils.save_image( ( (imBatch )**(1.0/2.2) ).data,
                '{0}/{1}_im.png'.format(opt.testRoot, j) )
        depthOut = 1 / torch.clamp(depthBatch + 1, 1e-6, 10) * segAllBatch.expand_as(depthBatch)
        vutils.save_image( ( depthOut*segAllBatch.expand_as(depthBatch) ).data,
                '{0}/{1}_depthGt.png'.format(opt.testRoot, j) )

        # Save the predicted results
        for n in range(0, len(albedoPreds) ):
            vutils.save_image( ( (albedoPreds[n] ) ** (1.0/2.2) ).data,
                    '{0}/{1}_albedoPred_{2}.png'.format(opt.testRoot, j, opt.cascadeLevel) )
        for n in range(0, len(normalPreds) ):
            vutils.save_image( ( 0.5*(normalPreds[n] + 1) ).data,
                    '{0}/{1}_normalPred_{2}.png'.format(opt.testRoot, j, opt.cascadeLevel) )
        for n in range(0, len(roughPreds) ):
            vutils.save_image( ( 0.5*(roughPreds[n] + 1) ).data,
                    '{0}/{1}_roughPred_{2}.png'.format(opt.testRoot, j, opt.cascadeLevel) )
        for n in range(0, len(depthPreds) ):
            depthOut = 1 / torch.clamp(depthPreds[n] + 1, 1e-6, 10) * segAllBatch.expand_as(depthPreds[n])
            vutils.save_image( ( depthOut * segAllBatch.expand_as(depthPreds[n]) ).data,
                    '{0}/{1}_depthPred_{2}.png'.format(opt.testRoot, j, n) )

testingLog.close()

# Save the error record
np.save('{0}/albedoError_{1}.npy'.format(opt.testRoot, epoch), albedoErrsNpList )
np.save('{0}/normalError_{1}.npy'.format(opt.testRoot, epoch), normalErrsNpList )
np.save('{0}/roughError_{1}.npy'.format(opt.testRoot, epoch), roughErrsNpList )
np.save('{0}/depthError_{1}.npy'.format(opt.testRoot, epoch), depthErrsNpList )
