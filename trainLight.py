import torch
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import argparse
import random
import os
import models
import torchvision.utils as vutils
import utils
import dataLoader
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import wrapperBRDFLight as wcg

parser = argparse.ArgumentParser()
# The locationi of training set
parser.add_argument('--dataRoot', default=None, help='path to input images')
parser.add_argument('--experimentBRDF', default=None, help='path to the model for BRDF prediction')
parser.add_argument('--experiment', default=None, help='the path to store samples and models')
# The basic training setting
parser.add_argument('--nepochBRDF0', type=int, default=14, help='the number of epochs for BRDF prediction')
parser.add_argument('--nepochBRDF1', type=int, default=7, help='the number of epochs for BRDF prediction')
parser.add_argument('--nepoch0', type=int, default=14, help='the number of epochs for training')
parser.add_argument('--nepoch1', type=int, default=12, help='the number of epochs for training')

parser.add_argument('--batchSize0', type=int, default=5, help='input batch size' )
parser.add_argument('--batchSize1', type=int, default=5, help='input batch size' )

parser.add_argument('--imHeight0', type=int, default=240, help='the height / width of the input image to network')
parser.add_argument('--imWidth0', type=int, default=320, help='the height / width of the input image to network')
parser.add_argument('--imHeight1', type=int, default=240, help='the height / width of the input image to network')
parser.add_argument('--imWidth1', type=int, default=320, help='the height / width of the input image to network')

parser.add_argument('--offset', type=float, default=1, help='the offset for log error')

parser.add_argument('--envRow', type=int, default=120, help='the number of samples of envmaps in y direction')
parser.add_argument('--envCol', type=int, default=160, help='the number of samples of envmaps in x direction')
parser.add_argument('--envHeight', type=int, default=8, help='the size of envmaps in y direction')
parser.add_argument('--envWidth', type=int, default=16, help='the size of envmaps in x direction')
parser.add_argument('--SGNum', type=int, default=12, help='the number of spherical Gaussian lobe' )

parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--deviceIds', type=int, nargs='+', default=[0, 1, 2], help='the gpus used for training network')
# The training weight
parser.add_argument('--reconstWeight', type=float, default=10, help='the weight for reconstruction error' )
parser.add_argument('--renderWeight', type=float, default=1.0, help='the weight for the rendering' )
# Cascae Level
parser.add_argument('--cascadeLevel', type=int, default=0, help='the casacade level')

# The detail network setting
opt = parser.parse_args()
print(opt)

opt.gpuId = opt.deviceIds[0]
torch.multiprocessing.set_sharing_strategy('file_system')

if opt.offset is None:
    offset = 5.0
else:
    offset = opt.offset


if opt.experiment is None:
    opt.experiment = 'check_cascadeLight%d_sg%d_offset%.1f' \
            % (opt.cascadeLevel, opt.SGNum, opt.offset )

os.system('mkdir {0}'.format(opt.experiment) )
os.system('cp *.py %s' % opt.experiment )

recW, renW = opt.reconstWeight, opt.renderWeight

if opt.cascadeLevel == 0:
    opt.nepoch = opt.nepoch0
    opt.batchSize = opt.batchSize0
    opt.imHeight, opt.imWidth = opt.imHeight0, opt.imWidth0
    opt.nepochBRDF = opt.nepochBRDF0
elif opt.cascadeLevel == 1:
    opt.nepoch = opt.nepoch1
    opt.batchSize = opt.batchSize1
    opt.imHeight, opt.imWidth = opt.imHeight1, opt.imWidth1
    opt.nepochBRDF = opt.nepochBRDF1


if opt.experimentBRDF is None:
    opt.experimentBRDF = 'check_cascade%d_w%d_h%d' % (opt.cascadeLevel,
            opt.imWidth, opt.imHeight )

opt.seed = 0
print("Random Seed: ", opt.seed )
random.seed(opt.seed )
torch.manual_seed(opt.seed )

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# Initial Network
encoder = models.encoder0(cascadeLevel = opt.cascadeLevel )
albedoDecoder = models.decoder0(mode=0 )
normalDecoder = models.decoder0(mode=1 )
roughDecoder = models.decoder0(mode=2 )
depthDecoder = models.decoder0(mode=4 )

lightEncoder = models.encoderLight(cascadeLevel = opt.cascadeLevel, SGNum =
        opt.SGNum )
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
        torch.load('{0}/encoder{1}_{2}.pth'.format(opt.experimentBRDF, opt.cascadeLevel, opt.nepochBRDF-1) ).state_dict() )
for param in encoder.parameters():
    param.requires_grad = False

albedoDecoder.load_state_dict(
        torch.load('{0}/albedo{1}_{2}.pth'.format(opt.experimentBRDF, opt.cascadeLevel, opt.nepochBRDF-1) ).state_dict() )
for param in albedoDecoder.parameters():
    param.requires_grad = False

normalDecoder.load_state_dict(
        torch.load('{0}/normal{1}_{2}.pth'.format(opt.experimentBRDF, opt.cascadeLevel, opt.nepochBRDF-1) ).state_dict() )
for param in normalDecoder.parameters():
    param.requires_grad = False

roughDecoder.load_state_dict(
        torch.load('{0}/rough{1}_{2}.pth'.format(opt.experimentBRDF, opt.cascadeLevel, opt.nepochBRDF-1) ).state_dict() )
for param in roughDecoder.parameters():
    param.requires_grad = False

depthDecoder.load_state_dict(
        torch.load('{0}/depth{1}_{2}.pth'.format(opt.experimentBRDF, opt.cascadeLevel, opt.nepochBRDF-1) ).state_dict() )
for param in depthDecoder.parameters():
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
# Optimizer
lr_scale = 1
opLightEncoder = optim.Adam(lightEncoder.parameters(), lr=1e-4 * lr_scale, betas=(0.5, 0.999) )
opAxisDecoder = optim.Adam(axisDecoder.parameters(), lr=1e-4 * lr_scale, betas=(0.5, 0.999) )
opLambDecoder = optim.Adam(lambDecoder.parameters(), lr=1e-4 * lr_scale, betas=(0.5, 0.999) )
opWeightDecoder = optim.Adam(weightDecoder.parameters(), lr=1e-4 * lr_scale, betas=(0.5, 0.999) )
#####################################


####################################
brdfDataset = dataLoader.BatchLoader( opt.dataRoot, isAllLight = True,
        imWidth = opt.imWidth, imHeight = opt.imHeight, isLight = True,
        cascadeLevel = opt.cascadeLevel, SGNum = opt.SGNum )
brdfLoader = DataLoader(brdfDataset, batch_size = opt.batchSize, num_workers =
        16, shuffle = True )

j = 0
# BRDFLost
albedoErrsNpList = np.ones( [1, 1], dtype = np.float32 )
normalErrsNpList = np.ones( [1, 1], dtype = np.float32 )
roughErrsNpList= np.ones( [1, 1], dtype = np.float32 )
depthErrsNpList = np.ones( [1, 1], dtype = np.float32 )

reconstErrsNpList = np.ones( [1, 1], dtype = np.float32 )
renderErrsNpList = np.ones( [1, 1], dtype = np.float32 )
for epoch in list(range(0, opt.nepoch) ):
    trainingLog = open('{0}/trainingLog_{1}.txt'.format(opt.experiment, epoch), 'w')
    for i, dataBatch in enumerate(brdfLoader):
        j += 1

        #####################################################################################################################
        ############################################# Train with CGBRDF dataset #############################################
        #####################################################################################################################
        # Clear the gradient in optimizer
        opLightEncoder.zero_grad()
        opAxisDecoder.zero_grad()
        opLambDecoder.zero_grad()
        opWeightDecoder.zero_grad()

        albedoPair, normalPair, roughPair, depthPair,  \
        envmapsPair, renderPair \
        = wcg.wrapperBRDFLight(dataBatch, opt, encoder, \
        albedoDecoder, normalDecoder, roughDecoder, depthDecoder, \
        lightEncoder, axisDecoder, lambDecoder, weightDecoder, \
        output2env, renderLayer, offset )

        albedoPred, albedoErr = albedoPair[0], albedoPair[1]
        albedoBatch = albedoPair[2]
        normalPred, normalErr = normalPair[0], normalPair[1]
        normalBatch = normalPair[2]
        roughPred, roughErr = roughPair[0], roughPair[1]
        roughBatch = roughPair[2]
        depthPred, depthErr = depthPair[0], depthPair[1]
        depthBatch = depthPair[2]
        envmapsPredScaledImage, reconstErr = envmapsPair[0], envmapsPair[1]
        envmapsBatch = envmapsPair[2]
        renderedImPred, renderErr = renderPair[0], renderPair[1]
        imBatch = renderPair[2]


        # Back propagate the gradients
        totalErr = renW * renderErr + recW * reconstErr
        totalErr.backward()

        # Update the network parameter
        opLightEncoder.step()
        opAxisDecoder.step()
        opLambDecoder.step()
        opWeightDecoder.step()

        # Output training error
        utils.writeErrToScreen('albedo', [albedoErr], epoch, j)
        utils.writeErrToScreen('normal', [normalErr], epoch, j)
        utils.writeErrToScreen('rough', [roughErr], epoch, j)
        utils.writeErrToScreen('depth', [depthErr], epoch, j)

        utils.writeErrToScreen('reconstErrors', [reconstErr], epoch, j)
        utils.writeErrToScreen('renderErrors', [renderErr], epoch, j)

        utils.writeErrToFile('albedo', [albedoErr], trainingLog, epoch, j)
        utils.writeErrToFile('normal', [normalErr], trainingLog, epoch, j)
        utils.writeErrToFile('rough', [roughErr], trainingLog, epoch, j)
        utils.writeErrToFile('depth', [depthErr], trainingLog, epoch, j)

        utils.writeErrToFile('reconstErrors', [reconstErr], trainingLog, epoch, j)
        utils.writeErrToFile('renderErrors', [renderErr], trainingLog, epoch, j)

        albedoErrsNpList = np.concatenate( [albedoErrsNpList, utils.turnErrorIntoNumpy( [albedoErr] )], axis=0)
        normalErrsNpList = np.concatenate( [normalErrsNpList, utils.turnErrorIntoNumpy( [normalErr] )], axis=0)
        roughErrsNpList = np.concatenate( [roughErrsNpList, utils.turnErrorIntoNumpy( [roughErr] )], axis=0)
        depthErrsNpList = np.concatenate( [depthErrsNpList, utils.turnErrorIntoNumpy( [depthErr] )], axis=0)

        reconstErrsNpList = np.concatenate( [reconstErrsNpList, utils.turnErrorIntoNumpy( [reconstErr] )], axis=0 )
        renderErrsNpList = np.concatenate( [renderErrsNpList, utils.turnErrorIntoNumpy( [renderErr] )], axis=0 )

        if j < 1000:
            utils.writeNpErrToScreen('albedoAccu', np.mean(albedoErrsNpList[1:j+1, :], axis=0), epoch, j )
            utils.writeNpErrToScreen('normalAccu', np.mean(normalErrsNpList[1:j+1, :], axis=0), epoch, j )
            utils.writeNpErrToScreen('roughAccu', np.mean(roughErrsNpList[1:j+1, :], axis=0), epoch, j )
            utils.writeNpErrToScreen('depthAccu', np.mean(depthErrsNpList[1:j+1, :], axis=0), epoch, j )

            utils.writeNpErrToScreen('reconstAccu', np.mean(reconstErrsNpList[1:j+1, :], axis=0), epoch, j )
            utils.writeNpErrToScreen('renderAccu', np.mean(renderErrsNpList[1:j+1, :], axis=0), epoch, j )

            utils.writeNpErrToFile('albedoAccu', np.mean(albedoErrsNpList[1:j+1, :], axis=0), trainingLog, epoch, j)
            utils.writeNpErrToFile('normalAccu', np.mean(normalErrsNpList[1:j+1, :], axis=0), trainingLog, epoch, j)
            utils.writeNpErrToFile('roughAccu', np.mean(roughErrsNpList[1:j+1, :], axis=0), trainingLog, epoch, j)
            utils.writeNpErrToFile('depthAccu', np.mean(depthErrsNpList[1:j+1, :], axis=0), trainingLog, epoch, j)

            utils.writeNpErrToFile('reconstAccu', np.mean(reconstErrsNpList[1:j+1, :], axis=0), trainingLog, epoch, j )
            utils.writeNpErrToFile('renderAccu', np.mean(renderErrsNpList[1:j+1, :], axis=0), trainingLog, epoch, j )
        else:
            utils.writeNpErrToScreen('albedoAccu', np.mean(albedoErrsNpList[j-999:j+1, :], axis=0), epoch, j)
            utils.writeNpErrToScreen('normalAccu', np.mean(normalErrsNpList[j-999:j+1, :], axis=0), epoch, j)
            utils.writeNpErrToScreen('roughAccu', np.mean(roughErrsNpList[j-999:j+1, :], axis=0), epoch, j)
            utils.writeNpErrToScreen('depthAccu', np.mean(depthErrsNpList[j-999:j+1, :], axis=0), epoch, j)

            utils.writeNpErrToScreen('reconstAccu', np.mean(reconstErrsNpList[j-999:j+1, :], axis=0), epoch, j )
            utils.writeNpErrToScreen('renderAccu', np.mean(renderErrsNpList[j-999:j+1, :], axis=0), epoch, j )

            utils.writeNpErrToFile('albedoAccu', np.mean(albedoErrsNpList[j-999:j+1, :], axis=0), trainingLog, epoch, j)
            utils.writeNpErrToFile('normalAccu', np.mean(normalErrsNpList[j-999:j+1, :], axis=0), trainingLog, epoch, j)
            utils.writeNpErrToFile('roughAccu', np.mean(roughErrsNpList[j-999:j+1, :], axis=0), trainingLog, epoch, j)
            utils.writeNpErrToFile('depthAccu', np.mean(depthErrsNpList[j-999:j+1, :], axis=0), trainingLog, epoch, j)

            utils.writeNpErrToFile('reconstAccu', np.mean(reconstErrsNpList[j-999:j+1, :], axis=0), trainingLog, epoch, j )
            utils.writeNpErrToFile('renderAccu', np.mean(renderErrsNpList[j-999:j+1, :], axis=0), trainingLog, epoch, j )


        if j == 1 or j% 2000 == 0:
            # Save the groundtruth results
            vutils.save_image( ( (albedoBatch ) ** (1.0/2.2) ).data,
                    '{0}/{1}_albedoGt_{2}.png'.format(opt.experiment, j, 0) )
            vutils.save_image( ( 0.5*(normalBatch + 1) ).data,
                    '{0}/{1}_normalGt_{2}.png'.format(opt.experiment, j, 0) )
            vutils.save_image( ( 0.5*(roughBatch + 1) ).data,
                    '{0}/{1}_roughGt_{2}.png'.format(opt.experiment, j, 0) )
            depthOutGt = 1 / torch.clamp(depthBatch + 1, 1e-6, 10)
            vutils.save_image( ( depthOutGt ).data,
                    '{0}/{1}_depthGt_{2}.png'.format(opt.experiment, j, 0) )

            vutils.save_image( ( (imBatch )**(1.0/2.2) ).data,
                '{0}/{1}_im.png'.format(opt.experiment, j) )

            utils.writeEnvToFile(envmapsBatch, 0, '{0}/{1}_envmapGt.png'.format(opt.experiment, j) )

            # Save the predicted results
            vutils.save_image( ( (albedoPred ) ** (1.0/2.2) ).data,
                    '{0}/{1}_albedoPred_{2}.png'.format(opt.experiment, j, 0) )
            vutils.save_image( ( 0.5*(normalPred + 1) ).data,
                    '{0}/{1}_normalPred_{2}.png'.format(opt.experiment, j, 0) )
            vutils.save_image( ( 0.5*(roughPred + 1) ).data,
                    '{0}/{1}_roughPred_{2}.png'.format(opt.experiment, j, 0) )
            depthOut = 1 / torch.clamp(depthPred + 1, 1e-6, 10)
            vutils.save_image( ( depthOut ).data,
                    '{0}/{1}_depthPred_{2}.png'.format(opt.experiment, j, 0) )

            vutils.save_image( ( (renderedImPred )**(1.0/2.2) ).data,
                '{0}/{1}_imRendered.png'.format(opt.experiment, j) )

            utils.writeEnvToFile(envmapsPredScaledImage, 0, '{0}/{1}_envmapPred.png'.format(opt.experiment, j) )

    trainingLog.close()

    # Update the training rate
    if (epoch + 1) % 10 == 0:
        for param_group in opLightEncoder.param_groups:
            param_group['lr'] /= 2
        for param_group in opAxisDecoder.param_groups:
            param_group['lr'] /= 2
        for param_group in opLambDecoder.param_groups:
            param_group['lr'] /= 2
        for param_group in opWeightDecoder.param_groups:
            param_group['lr'] /= 2

    # Save the error record
    np.save('{0}/albedoError_{1}.npy'.format(opt.experiment, epoch), albedoErrsNpList )
    np.save('{0}/normalError_{1}.npy'.format(opt.experiment, epoch), normalErrsNpList )
    np.save('{0}/roughError_{1}.npy'.format(opt.experiment, epoch), roughErrsNpList )
    np.save('{0}/depthError_{1}.npy'.format(opt.experiment, epoch), depthErrsNpList )

    np.save('{0}/reconstError_{1}.npy'.format(opt.experiment, epoch), reconstErrsNpList )
    np.save('{0}/renderError_{1}.npy'.format(opt.experiment, epoch), renderErrsNpList )

    # save the models
    torch.save(lightEncoder.module, '{0}/lightEncoder{1}_{2}.pth'.format(opt.experiment, opt.cascadeLevel, epoch ) )
    torch.save(axisDecoder.module, '{0}/axisDecoder{1}_{2}.pth'.format(opt.experiment, opt.cascadeLevel, epoch ) )
    torch.save(lambDecoder.module, '{0}/lambDecoder{1}_{2}.pth'.format(opt.experiment, opt.cascadeLevel, epoch ) )
    torch.save(weightDecoder.module, '{0}/weightDecoder{1}_{2}.pth'.format(opt.experiment, opt.cascadeLevel, epoch ) )
