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
import nyuDataLoader as dataLoader_nyu
import dataLoader as dataLoader_ours
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import wrapperBRDF as wcg
import wrapperNYU as wnyu
import scipy.io as io
import os.path as osp

parser = argparse.ArgumentParser()
# The locationi of training set
parser.add_argument('--dataRoot', default=None, help='path to input images')
parser.add_argument('--NYURoot', default=None, help='path to the NYU dataset')
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
parser.add_argument('--albedoWeight', type=float, default=0.75, help='the weight for the diffuse component')
parser.add_argument('--normalWeight', type=float, default=0.5, help='the weight for the diffuse component')
parser.add_argument('--roughWeight', type=float, default=0.25, help='the weight for the roughness component')
parser.add_argument('--depthWeight', type=float, default=0.25, help='the weight for depth component')
# The training weight on NYU
parser.add_argument('--normalNYUWeight', type=float, default=4.5, help='the weight for the diffuse component')
parser.add_argument('--depthNYUWeight', type=float, default=4.5, help='the weight for depth component')
# Cascae Level
parser.add_argument('--cascadeLevel', type=int, default=0, help='the casacade level')

# The detail network setting
opt = parser.parse_args()
print(opt)

opt.gpuId = opt.deviceIds[0]
torch.multiprocessing.set_sharing_strategy('file_system')

if opt.experiment is None:
    opt.experiment = 'check_cascadeNYU%d' % opt.cascadeLevel

os.system('mkdir {0}'.format(opt.experiment) )
os.system('cp *.py %s' % opt.experiment )

if opt.experimentBRDF is None:
    opt.experimentBRDF = 'check_cascade0_w%d_h%d' % (opt.imWidth, opt.imHeight )

albeW, normW = opt.albedoWeight, opt.normalWeight
rougW = opt.roughWeight
deptW = opt.depthWeight

normNYUW = opt.normalNYUWeight
depthNYUW = opt.depthNYUWeight

opt.seed = 0
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
####################################################################


#########################################
encoder.load_state_dict( torch.load('{0}/encoder{1}_{2}.pth'.format(opt.experimentBRDF,
            opt.cascadeLevel, opt.nepochBRDF-1 ) ).state_dict() )
albedoDecoder.load_state_dict( torch.load('{0}/albedo{1}_{2}.pth'.format(opt.experimentBRDF,
            opt.cascadeLevel, opt.nepochBRDF-1 ) ).state_dict() )
normalDecoder.load_state_dict( torch.load('{0}/normal{1}_{2}.pth'.format(opt.experimentBRDF,
            opt.cascadeLevel, opt.nepochBRDF-1 ) ).state_dict() )
roughDecoder.load_state_dict( torch.load('{0}/rough{1}_{2}.pth'.format(opt.experimentBRDF,
            opt.cascadeLevel, opt.nepochBRDF-1 ) ).state_dict() )
depthDecoder.load_state_dict( torch.load('{0}/depth{1}_{2}.pth'.format(opt.experimentBRDF,
            opt.cascadeLevel, opt.nepochBRDF-1 ) ).state_dict() )

lr_scale = 0.5

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
NYUDataset = dataLoader_nyu.NYULoader(
        imRoot = osp.join(opt.NYURoot, 'images'),
        normalRoot = osp.join(opt.NYURoot, 'normals'),
        depthRoot = osp.join(opt.NYURoot, 'depths'),
        segRoot = osp.join(opt.NYURoot, 'masks'),
        imHeight = opt.imHeight,
        imWidth = opt.imWidth,
        phase = 'TRAIN' )
trainDataset = dataLoader_nyu.ConcatDataset(brdfDataset, NYUDataset)
brdfLoader = DataLoader(trainDataset, batch_size = opt.batchSize, num_workers =
        6, shuffle = True)

j = 0
# BRDFLost
albedoErrsNpList = np.ones( [1, 1], dtype = np.float32 )
normalErrsNpList = np.ones( [1, 1], dtype = np.float32 )
roughErrsNpList= np.ones( [1, 1], dtype = np.float32 )
depthErrsNpList = np.ones( [1, 1], dtype = np.float32 )

normalNYUErrsNpList = np.ones([1, 1], dtype=np.float32 )
angleNYUErrsNpList = np.ones([1, 1], dtype = np.float32 )
depthNYUErrsNpList = np.ones([1, 1], dtype=np.float32 )

for epoch in list(range(0, opt.nepoch) ):
    trainingLog = open('{0}/trainingLog_{1}.txt'.format(opt.experiment, epoch), 'w')
    for i, trainBatch in enumerate(brdfLoader):
        j += 1
        dataBatch = trainBatch[0]
        NYUBatch = trainBatch[1]

        #####################################################################################################################
        ############################################# Train with CGBRDF dataset #############################################
        #####################################################################################################################
        # Clear the gradient in optimizer
        opEncoder.zero_grad()
        opAlbedo.zero_grad()
        opNormal.zero_grad()
        opRough.zero_grad()
        opDepth.zero_grad()

        albedoPair, normalPair, roughPair, depthPair \
        = wcg.wrapperBRDF(dataBatch, opt, encoder,
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
        ######################################## Train with NYU dataset ##############################################
        ##############################################################################################################
        # Clear the gradient in optimizer
        opEncoder.zero_grad()
        opAlbedo.zero_grad()
        opNormal.zero_grad()
        opRough.zero_grad()
        opDepth.zero_grad()

        albedoPair, normalPair, roughPair, depthPair \
        = wnyu.wrapperNYU(NYUBatch, opt, encoder,
                albedoDecoder, normalDecoder, roughDecoder, depthDecoder )

        albedoPred = albedoPair[0]
        normalPred, normalErr, angleErr = normalPair[0], normalPair[1], normalPair[2]
        roughPred = roughPair[0]
        depthPred, depthErr = depthPair[0], depthPair[1]

        totalErr = normNYUW * normalErr + depthNYUW * depthErr
        totalErr.backward()

        # Update the network parameter
        opEncoder.step()
        opAlbedo.step()
        opNormal.step()
        opRough.step()
        opDepth.step()

        # Output training error
        utils.writeErrToScreen('normalNYU', [normalErr], epoch, j)
        utils.writeErrToScreen('angleNYU', [angleErr], epoch, j)
        utils.writeErrToScreen('depthNYU', [depthErr], epoch, j)
        utils.writeErrToFile('normalNYU', [normalErr], trainingLog, epoch, j)
        utils.writeErrToFile('angleNYU', [angleErr], trainingLog, epoch, j)
        utils.writeErrToFile('depthNYU', [depthErr], trainingLog, epoch, j)
        normalNYUErrsNpList = np.concatenate( [normalNYUErrsNpList, utils.turnErrorIntoNumpy( [normalErr] )], axis=0)
        angleNYUErrsNpList = np.concatenate( [angleNYUErrsNpList, utils.turnErrorIntoNumpy( [angleErr] )], axis=0)
        depthNYUErrsNpList = np.concatenate( [depthNYUErrsNpList, utils.turnErrorIntoNumpy( [depthErr] )], axis=0)

        if j < 1000:
            utils.writeNpErrToScreen('normalAccuNYU', np.mean(normalNYUErrsNpList[1:j+1, :], axis=0), epoch, j)
            utils.writeNpErrToScreen('angleAccuNYU', np.mean(angleNYUErrsNpList[1:j+1, :], axis=0), epoch, j)
            utils.writeNpErrToScreen('depthAccuNYU', np.mean(depthNYUErrsNpList[1:j+1, :], axis=0), epoch, j)

            utils.writeNpErrToFile('normalAccuNYU', np.mean(normalNYUErrsNpList[1:j+1, :], axis=0), trainingLog, epoch, j)
            utils.writeNpErrToFile('angleAccuNYU', np.mean(angleNYUErrsNpList[1:j+1, :], axis=0), trainingLog, epoch, j)
            utils.writeNpErrToFile('depthAccuNYU', np.mean(depthNYUErrsNpList[1:j+1, :], axis=0), trainingLog, epoch, j)
        else:
            utils.writeNpErrToScreen('normalAccuNYU', np.mean(normalNYUErrsNpList[j-999:j+1, :], axis=0), epoch, j)
            utils.writeNpErrToScreen('angleAccuNYU', np.mean(angleNYUErrsNpList[j-999:j+1, :], axis=0), epoch, j)
            utils.writeNpErrToScreen('depthAccuNYU', np.mean(depthNYUErrsNpList[j-999:j+1, :], axis=0), epoch, j)

            utils.writeNpErrToFile('normalAccuNYU', np.mean(normalNYUErrsNpList[j-999:j+1, :], axis=0), trainingLog, epoch, j)
            utils.writeNpErrToFile('angleAccuNYU', np.mean(angleNYUErrsNpList[j-999:j+1, :], axis=0), trainingLog, epoch, j)
            utils.writeNpErrToFile('depthAccuNYU', np.mean(depthNYUErrsNpList[j-999:j+1, :], axis=0), trainingLog, epoch, j)

        if j == 1 or j% 500 == 0:
            # Save the predicted results
            vutils.save_image( ( (albedoPred ) ** (1.0/2.2) ).data,
                    '{0}/{1}_albedoPredNYU_{2}.png'.format(opt.experiment, j, 0) )

            vutils.save_image( ( 0.5*(normalPred + 1) ).data,
                    '{0}/{1}_normalPredNYU_{2}.png'.format(opt.experiment, j, 0) )

            vutils.save_image( ( 0.5*(roughPred + 1) ).data,
                    '{0}/{1}_roughPredNYU_{2}.png'.format(opt.experiment, j, 0) )

            depthOut = 1 / torch.clamp(depthPred + 1, 1e-6, 10)
            vutils.save_image( ( depthOut ).data,
                    '{0}/{1}_depthPredNYU_{2}.png'.format(opt.experiment, j, 0) )

        if j % 2000 == 0:
            # save the models
            torch.save(encoder.module, '{0}/encoder{1}_{2}_{3}.pth'.format(opt.experiment, opt.cascadeLevel, epoch, j) )
            torch.save(albedoDecoder.module, '{0}/albedo{1}_{2}_{3}.pth'.format(opt.experiment, opt.cascadeLevel, epoch, j) )
            torch.save(normalDecoder.module, '{0}/normal{1}_{2}_{3}.pth'.format(opt.experiment, opt.cascadeLevel, epoch, j) )
            torch.save(roughDecoder.module, '{0}/rough{1}_{2}_{3}.pth'.format(opt.experiment, opt.cascadeLevel, epoch, j) )
            torch.save(depthDecoder.module, '{0}/depth{1}_{2}_{3}.pth'.format(opt.experiment, opt.cascadeLevel, epoch, j) )
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

    np.save('{0}/normalNYUError_{1}.npy'.format(opt.experiment, epoch), normalNYUErrsNpList )
    np.save('{0}/angleNYUError_{1}.npy'.format(opt.experiment, epoch), angleNYUErrsNpList )

    # save the models
    torch.save(encoder.module, '{0}/encoder{1}_{2}.pth'.format(opt.experiment, opt.cascadeLevel, epoch) )
    torch.save(albedoDecoder.module, '{0}/albedo{1}_{2}.pth'.format(opt.experiment, opt.cascadeLevel, epoch) )
    torch.save(normalDecoder.module, '{0}/normal{1}_{2}.pth'.format(opt.experiment, opt.cascadeLevel, epoch) )
    torch.save(roughDecoder.module, '{0}/rough{1}_{2}.pth'.format(opt.experiment, opt.cascadeLevel, epoch) )
    torch.save(depthDecoder.module, '{0}/depth{1}_{2}.pth'.format(opt.experiment, opt.cascadeLevel, epoch) )
