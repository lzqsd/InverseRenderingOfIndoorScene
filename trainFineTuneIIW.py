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
import iiwDataLoader as dataLoader_iiw
import dataLoader as dataLoader_ours
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import wrapperBRDF as wcg
import wrapperIIW as wiiw
import scipy.io as io

parser = argparse.ArgumentParser()
# The locationi of training set
parser.add_argument('--dataRoot', default=None, help='path to input images')
parser.add_argument('--IIWRoot', default=None, help='path to the IIW dataset')
parser.add_argument('--experimentBRDF', default=None, help='path to the model for BRDF prediction')
parser.add_argument('--experiment', default=None, help='the path to store samples and models')
# The basic training setting
parser.add_argument('--nepochBRDF', type=int, default=14, help='the number of epochs for BRDF prediction')
parser.add_argument('--nepoch', type=int, default=2, help='the number of epochs for training')
parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
parser.add_argument('--imHeight', type=int, default=240, help='the height / width of the input image to network')
parser.add_argument('--imWidth', type=int, default=320, help='the height / width of the input image to network')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--deviceIds', type=int, nargs='+', default=[0, 1], help='the gpus used for training network')
# The training weight
parser.add_argument('--albedoWeight', type=float, default=1.5, help='the weight for the diffuse component')
parser.add_argument('--normalWeight', type=float, default=1.0, help='the weight for the diffuse component')
parser.add_argument('--roughWeight', type=float, default=0.5, help='the weight for the roughness component')
parser.add_argument('--depthWeight', type=float, default=0.5, help='the weight for depth component')
# The training weight on IIW
parser.add_argument('--rankWeight', type=float, default=2.0, help='the weight of ranking')
# Cascae Level
parser.add_argument('--cascadeLevel', type=int, default=0, help='the casacade level')


# The detail network setting
opt = parser.parse_args()
print(opt)

opt.gpuId = opt.deviceIds[0]
torch.multiprocessing.set_sharing_strategy('file_system')

if opt.experiment is None:
    opt.experiment = 'check_cascadeIIW0'

os.system('mkdir {0}'.format(opt.experiment) )
os.system('cp *.py %s' % opt.experiment )

if opt.experimentBRDF is None:
    opt.experimentBRDF = 'check_cascade0_w%d_h%d' % (opt.imWidth, opt.imHeight )

albeW, normW = opt.albedoWeight, opt.normalWeight
rougW = opt.roughWeight
deptW = opt.depthWeight

rankW = opt.rankWeight

opt.seed = 0
print("Random Seed: ", opt.seed )
random.seed(opt.seed )
torch.manual_seed(opt.seed )

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

####################################
# Initial Network
encoder = models.encoder0(cascadeLevel = 0 )
albedoDecoder = models.decoder0(mode=0 )
normalDecoder = models.decoder0(mode=1 )
roughDecoder = models.decoder0(mode=2 )
depthDecoder = models.decoder0(mode=4 )

#########################################
encoder.load_state_dict(torch.load('{0}/encoder{1}_{2}.pth'.format(opt.experimentBRDF,
    0, opt.nepochBRDF-1) ).state_dict() )
albedoDecoder.load_state_dict(torch.load('{0}/albedo{1}_{2}.pth'.format(opt.experimentBRDF,
    0, opt.nepochBRDF-1) ).state_dict() )
normalDecoder.load_state_dict(torch.load('{0}/normal{1}_{2}.pth'.format(opt.experimentBRDF,
    0, opt.nepochBRDF-1) ).state_dict() )
roughDecoder.load_state_dict(torch.load('{0}/rough{1}_{2}.pth'.format(opt.experimentBRDF,
    0, opt.nepochBRDF-1) ).state_dict() )
depthDecoder.load_state_dict(torch.load('{0}/depth{1}_{2}.pth'.format(opt.experimentBRDF,
    0, opt.nepochBRDF-1) ).state_dict() )
lr_scale = 1
#########################################
encoder = nn.DataParallel(encoder, device_ids = opt.deviceIds )
albedoDecoder = nn.DataParallel(albedoDecoder, device_ids = opt.deviceIds )
normalDecoder = nn.DataParallel(normalDecoder, device_ids = opt.deviceIds )
roughDecoder = nn.DataParallel(roughDecoder, device_ids = opt.deviceIds )
depthDecoder = nn.DataParallel(depthDecoder, device_ids = opt.deviceIds )

##############  ######################
# Send things into GPU
if opt.cuda:
    encoder = encoder.cuda(opt.gpuId )
    albedoDecoder = albedoDecoder.cuda(opt.gpuId )
    normalDecoder = normalDecoder.cuda(opt.gpuId )
    roughDecoder = roughDecoder.cuda(opt.gpuId )
    depthDecoder = depthDecoder.cuda(opt.gpuId )
####################################


####################################
# Optimizer
opEncoder = optim.Adam(encoder.parameters(), lr=1e-4 * lr_scale, betas=(0.5, 0.999) )
opAlbedo = optim.Adam(albedoDecoder.parameters(), lr=1e-4 * lr_scale, betas=(0.5, 0.999) )
opNormal = optim.Adam(normalDecoder.parameters(), lr=1e-4 * lr_scale, betas=(0.5, 0.999) )
opRough = optim.Adam(roughDecoder.parameters(), lr=1e-4 * lr_scale, betas=(0.5, 0.999) )
opDepth = optim.Adam(depthDecoder.parameters(), lr=1e-4 * lr_scale, betas=(0.5, 0.999) )
#####################################


####################################
brdfDataset = dataLoader_ours.BatchLoader( opt.dataRoot, imWidth = opt.imWidth, imHeight = opt.imHeight,
        cascadeLevel = 0, isLight = False )
IIWDataset = dataLoader_iiw.IIWLoader(
        dataRoot = opt.IIWRoot,
        imHeight = opt.imHeight,
        imWidth = opt.imWidth,
        phase = 'TRAIN' )
trainDataset = dataLoader_iiw.ConcatDataset(brdfDataset, IIWDataset)
brdfLoader = DataLoader(trainDataset, batch_size = opt.batchSize, num_workers =
        6, shuffle = True )

j = 0
# BRDFLost
albedoErrsNpList = np.ones( [1, 1], dtype = np.float32 )
normalErrsNpList = np.ones( [1, 1], dtype = np.float32 )
roughErrsNpList= np.ones( [1, 1], dtype = np.float32 )
depthErrsNpList = np.ones( [1, 1], dtype = np.float32 )

eqErrsNpList = np.ones([1, 1], dtype=np.float32 )
darkerErrsNpList = np.ones([1, 1], dtype=np.float32 )

for epoch in list(range(0, opt.nepoch) ):
    trainingLog = open('{0}/trainingLog_{1}.txt'.format(opt.experiment, epoch), 'w')
    for i, trainBatch in enumerate(brdfLoader):
        j += 1
        dataBatch = trainBatch[0]
        IIWBatch = trainBatch[1]

        #####################################################################################################################
        ############################################# Train with CGBRDF dataset #############################################
        #####################################################################################################################
        # Clear the gradient in optimizer
        opEncoder.zero_grad()
        opAlbedo.zero_grad()
        opNormal.zero_grad()
        opRough.zero_grad()
        opDepth.zero_grad()

        albedoPair, normalPair, roughPair, depthPair,  \
        = wcg.wrapperBRDF(dataBatch, opt, encoder, \
        albedoDecoder, normalDecoder, roughDecoder, depthDecoder )

        albedoPred, albedoErr = albedoPair[0], albedoPair[1]
        normalPred, normalErr = normalPair[0], normalPair[1]
        roughPred, roughErr = roughPair[0], roughPair[1]
        depthPred, depthErr = depthPair[0], depthPair[1]

        # Back propagate the gradients
        totalErr = 4 * albeW * albedoErr + normW * normalErr \
                + rougW *roughErr + deptW * depthErr
        totalErr.backward()

        # Update the network parameter
        opEncoder.step()
        opAlbedo.step()
        opNormal.step()
        opRough.step()
        opDepth.step()

        # Output training error
        utils.writeErrToScreen('albedo', [albedoErr], epoch, j)
        utils.writeErrToScreen('normal', [normalErr], epoch, j)
        utils.writeErrToScreen('rough', [roughErr], epoch, j)
        utils.writeErrToScreen('depth', [depthErr], epoch, j)

        utils.writeErrToFile('albedo', [albedoErr], trainingLog, epoch, j)
        utils.writeErrToFile('normal', [normalErr], trainingLog, epoch, j)
        utils.writeErrToFile('rough', [roughErr], trainingLog, epoch, j)
        utils.writeErrToFile('depth', [depthErr], trainingLog, epoch, j)

        albedoErrsNpList = np.concatenate( [albedoErrsNpList, utils.turnErrorIntoNumpy( [albedoErr] )], axis=0)
        normalErrsNpList = np.concatenate( [normalErrsNpList, utils.turnErrorIntoNumpy( [normalErr] )], axis=0)
        roughErrsNpList = np.concatenate( [roughErrsNpList, utils.turnErrorIntoNumpy( [roughErr] )], axis=0)
        depthErrsNpList = np.concatenate( [depthErrsNpList, utils.turnErrorIntoNumpy( [depthErr] )], axis=0)

        if j < 1000:
            utils.writeNpErrToScreen('albedoAccu', np.mean(albedoErrsNpList[1:j+1, :], axis=0), epoch, j )
            utils.writeNpErrToScreen('normalAccu', np.mean(normalErrsNpList[1:j+1, :], axis=0), epoch, j )
            utils.writeNpErrToScreen('roughAccu', np.mean(roughErrsNpList[1:j+1, :], axis=0), epoch, j )
            utils.writeNpErrToScreen('depthAccu', np.mean(depthErrsNpList[1:j+1, :], axis=0), epoch, j )

            utils.writeNpErrToFile('albedoAccu', np.mean(albedoErrsNpList[1:j+1, :], axis=0), trainingLog, epoch, j)
            utils.writeNpErrToFile('normalAccu', np.mean(normalErrsNpList[1:j+1, :], axis=0), trainingLog, epoch, j)
            utils.writeNpErrToFile('roughAccu', np.mean(roughErrsNpList[1:j+1, :], axis=0), trainingLog, epoch, j)
            utils.writeNpErrToFile('depthAccu', np.mean(depthErrsNpList[1:j+1, :], axis=0), trainingLog, epoch, j)
        else:
            utils.writeNpErrToScreen('albedoAccu', np.mean(albedoErrsNpList[j-999:j+1, :], axis=0), epoch, j)
            utils.writeNpErrToScreen('normalAccu', np.mean(normalErrsNpList[j-999:j+1, :], axis=0), epoch, j)
            utils.writeNpErrToScreen('roughAccu', np.mean(roughErrsNpList[j-999:j+1, :], axis=0), epoch, j)
            utils.writeNpErrToScreen('depthAccu', np.mean(depthErrsNpList[j-999:j+1, :], axis=0), epoch, j)

            utils.writeNpErrToFile('albedoAccu', np.mean(albedoErrsNpList[j-999:j+1, :], axis=0), trainingLog, epoch, j)
            utils.writeNpErrToFile('normalAccu', np.mean(normalErrsNpList[j-999:j+1, :], axis=0), trainingLog, epoch, j)
            utils.writeNpErrToFile('roughAccu', np.mean(roughErrsNpList[j-999:j+1, :], axis=0), trainingLog, epoch, j)
            utils.writeNpErrToFile('depthAccu', np.mean(depthErrsNpList[j-999:j+1, :], axis=0), trainingLog, epoch, j)

        if j == 1 or j% 2000 == 0:
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


        ##############################################################################################################
        ######################################## Train with IIW dataset ##############################################
        ##############################################################################################################
        # Clear the gradient in optimizer
        opEncoder.zero_grad()
        opAlbedo.zero_grad()
        opNormal.zero_grad()
        opRough.zero_grad()
        opDepth.zero_grad()

        albedoPair, normalPair, roughPair, depthPair,  \
        eqPair, darkerPair \
        = wiiw.wrapperIIW(IIWBatch, opt, encoder, \
        albedoDecoder, normalDecoder, roughDecoder, depthDecoder )

        albedoPred = albedoPair[0]
        normalPred = normalPair[0]
        roughPred = roughPair[0]
        depthPred = depthPair[0]
        eq, eqErr = eqPair[0], eqPair[1]
        darker, darkerErr = darkerPair[0], darkerPair[1]

        totalErr = rankW * eqErr + rankW * darkerErr
        totalErr.backward()

        # Update the network parameter
        opEncoder.step()
        opAlbedo.step()
        opNormal.step()
        opRough.step()
        opDepth.step()

        # Output training error
        utils.writeErrToScreen('equalIIW', [eqErr], epoch, j)
        utils.writeErrToScreen('darkerIIW', [darkerErr], epoch, j)
        utils.writeErrToFile('equalIIW', [eqErr], trainingLog, epoch, j)
        utils.writeErrToFile('darkerIIW', [darkerErr], trainingLog, epoch, j)
        eqErrsNpList = np.concatenate( [eqErrsNpList, utils.turnErrorIntoNumpy( [eqErr] )], axis=0)
        darkerErrsNpList = np.concatenate( [darkerErrsNpList, utils.turnErrorIntoNumpy( [darkerErr] )], axis=0)

        if j < 1000:
            utils.writeNpErrToScreen('eqAccuIIW', np.mean(eqErrsNpList[1:j+1, :], axis=0), epoch, j)
            utils.writeNpErrToScreen('darkerAccuIIW', np.mean(darkerErrsNpList[1:j+1, :], axis=0), epoch, j)

            utils.writeNpErrToFile('eqAccuIIW', np.mean(eqErrsNpList[1:j+1, :], axis=0), trainingLog, epoch, j)
            utils.writeNpErrToFile('darkerAccuIIW', np.mean(darkerErrsNpList[1:j+1, :], axis=0), trainingLog, epoch, j)
        else:
            utils.writeNpErrToScreen('eqAccuIIW', np.mean(eqErrsNpList[j-999:j+1, :], axis=0), epoch, j)
            utils.writeNpErrToScreen('darkerAccuIIW', np.mean(darkerErrsNpList[j-999:j+1, :], axis=0), epoch, j)

            utils.writeNpErrToFile('eqAccuIIW', np.mean(eqErrsNpList[j-999:j+1, :], axis=0), trainingLog, epoch, j)
            utils.writeNpErrToFile('darkerAccuIIW', np.mean(darkerErrsNpList[j-999:j+1, :], axis=0), trainingLog, epoch, j)

        if j == 1 or j% 500 == 0:
            eq['point'] = eq['point'].numpy()
            eq['weight'] = eq['weight'].numpy()
            darker['point'] = darker['point'].numpy()
            darker['weight'] = darker['weight'].numpy()
            io.savemat('{0}/{1}_eq.mat'.format(opt.experiment, j, 0), eq )
            io.savemat('{0}/{1}_darker.mat'.format(opt.experiment, j, 0), darker )

            # Save the predicted results
            vutils.save_image( ( (albedoPred ) ** (1.0/2.2) ).data,
                    '{0}/{1}_albedoPredIIW_{2}.png'.format(opt.experiment, j, 0) )
            vutils.save_image( ( 0.5*(normalPred + 1) ).data,
                    '{0}/{1}_normalPredIIW_{2}.png'.format(opt.experiment, j, 0) )
            vutils.save_image( ( 0.5*(roughPred + 1) ).data,
                    '{0}/{1}_roughPredIIW_{2}.png'.format(opt.experiment, j, 0) )
            depthOut = 1 / torch.clamp(depthPred + 1, 1e-6, 10)
            vutils.save_image( ( depthOut ).data,
                    '{0}/{1}_depthPredIIW_{2}.png'.format(opt.experiment, j, 0) )

        if j % 2000 == 0:
            # save the models
            torch.save(encoder.module, '{0}/encoder{1}_{2}_{3}.pth'.format(opt.experiment, opt.cascadeLevel, epoch, j ) )
            torch.save(albedoDecoder.module, '{0}/albedo{1}_{2}_{3}.pth'.format(opt.experiment, opt.cascadeLevel, epoch, j ) )
            torch.save(normalDecoder.module, '{0}/normal{1}_{2}_{3}.pth'.format(opt.experiment, opt.cascadeLevel, epoch, j ) )
            torch.save(roughDecoder.module, '{0}/rough{1}_{2}_{3}.pth'.format(opt.experiment, opt.cascadeLevel, epoch, j ) )
            torch.save(depthDecoder.module, '{0}/depth{1}_{2}_{3}.pth'.format(opt.experiment, opt.cascadeLevel, epoch, j ) )

        ######################################################################################################################

    trainingLog.close()

    # Update the training rate
    if (epoch + 1) % 10 == 0:
        for param_group in opEncoder.param_groups:
            param_group['lr'] /= 2
        for param_group in opAlbedo.param_groups:
            param_group['lr'] /= 2
        for param_group in opNormal.param_groups:
            param_group['lr'] /= 2
        for param_group in opRough.param_groups:
            param_group['lr'] /= 2
        for param_group in opDepth.param_groups:
            param_group['lr'] /= 2

    # Save the error record
    np.save('{0}/albedoError_{1}.npy'.format(opt.experiment, epoch), albedoErrsNpList )
    np.save('{0}/normalError_{1}.npy'.format(opt.experiment, epoch), normalErrsNpList )
    np.save('{0}/roughError_{1}.npy'.format(opt.experiment, epoch), roughErrsNpList )
    np.save('{0}/depthError_{1}.npy'.format(opt.experiment, epoch), depthErrsNpList )


    # save the models
    torch.save(encoder.module, '{0}/encoder{1}_{2}.pth'.format(opt.experiment, opt.cascadeLevel, epoch) )
    torch.save(albedoDecoder.module, '{0}/albedo{1}_{2}.pth'.format(opt.experiment, opt.cascadeLevel, epoch) )
    torch.save(normalDecoder.module, '{0}/normal{1}_{2}.pth'.format(opt.experiment, opt.cascadeLevel, epoch) )
    torch.save(roughDecoder.module, '{0}/rough{1}_{2}.pth'.format(opt.experiment, opt.cascadeLevel, epoch) )
    torch.save(depthDecoder.module, '{0}/depth{1}_{2}.pth'.format(opt.experiment, opt.cascadeLevel, epoch) )
