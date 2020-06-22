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
parser.add_argument('--experimentBRDF0', default=None, help='path to the model for BRDF prediction')
parser.add_argument('--experimentLight0', default=None, help='path to the model for light prediction')
parser.add_argument('--experimentBRDF1', default=None, help='path to the model for BRDF prediction')
parser.add_argument('--experiment', default=None, help='the path to store samples and models')
# The basic training setting
parser.add_argument('--nepochBRDF0', type=int, default=2, help='the number of epochs for BRDF prediction')
parser.add_argument('--nepochLight0', type=int, default=10, help='the number of epochs for light prediction')
parser.add_argument('--nepochBRDF1', type=int, default=7, help='the number of epochs for BRDF prediction')
parser.add_argument('--nepoch', type=int, default=2, help='the number of epochs for training')

parser.add_argument('--batchSize', type=int, default=4, help='input batch size')

parser.add_argument('--imHeight0', type=int, default=240, help='the height / width of the input image to network')
parser.add_argument('--imWidth0', type=int, default=320, help='the height / width of the input image to network')
parser.add_argument('--imHeight1', type=int, default=240, help='the height / width of the input image to network')
parser.add_argument('--imWidth1', type=int, default=320, help='the height / width of the input image to network')

parser.add_argument('--envRow', type=int, default=120, help='the number of samples of envmaps in y direction')
parser.add_argument('--envCol', type=int, default=160, help='the number of samples of envmaps in x direction')
parser.add_argument('--envHeight', type=int, default=8, help='the size of envmaps in y direction')
parser.add_argument('--envWidth', type=int, default=16, help='the size of envmaps in x direction')
parser.add_argument('--SGNum', type=int, default=12, help='the number of spherical Gaussian lobe' )
parser.add_argument('--offset', type=float, default=1.0, help='the default offset when computing loss' )

parser.add_argument('--isFineTune', action='store_true', help='if we should fine tune the model')
parser.add_argument('--nepochFineTune', type=int, default=0, help='the model for fine tuning')

parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--deviceIds', type=int, nargs='+', default=[0, 1], help='the gpus used for training network')
# The training weight
parser.add_argument('--albedoWeight', type=float, default=1.5, help='the weight for the diffuse component')
parser.add_argument('--normalWeight', type=float, default=1.0, help='the weight for the diffuse component')
parser.add_argument('--roughWeight', type=float, default=0.5, help='the weight for the roughness component')
parser.add_argument('--depthWeight', type=float, default=0.5, help='the weight for depth component')
parser.add_argument('--rankWeight', type=float, default=2.0, help='the weight of ranking')
# Cascae Level
parser.add_argument('--cascadeLevel', type=int, default=1, help='the casacade level')

# The detail network setting
opt = parser.parse_args()
print(opt)

opt.gpuId = opt.deviceIds[0]
torch.multiprocessing.set_sharing_strategy('file_system')

if opt.experiment is None:
    opt.experiment = 'check_cascadeIIW1'
os.system('mkdir {0}'.format(opt.experiment) )
os.system('cp *.py %s' % opt.experiment )

if opt.experimentBRDF0 is None:
    opt.experimentBRDF0 = 'check_cascadeIIW0'

if opt.experimentLight0 is None:
    opt.experimentLight0 = 'check_cascadeLight%d_sg%d_offset%.1f' \
            % (0, opt.SGNum, opt.offset )

if opt.experimentBRDF1 is None:
    opt.experimentBRDF1 = 'check_cascade%d_w%d_h%d' \
            % (1, opt.imWidth1, opt.imHeight1 )

albeW, normW = opt.albedoWeight, opt.normalWeight
rougW = opt.roughWeight
deptW = opt.depthWeight

rankW = opt.rankWeight

opt.imHeight = opt.imHeight1
opt.imWidth = opt.imWidth1

opt.seed = 0
print("Random Seed: ", opt.seed )
random.seed(opt.seed )
torch.manual_seed(opt.seed )

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# Initial Network
encoder = models.encoder0(cascadeLevel = 0 )
albedoDecoder = models.decoder0(mode=0 )
normalDecoder = models.decoder0(mode=1 )
roughDecoder = models.decoder0(mode=2 )
depthDecoder = models.decoder0(mode=4 )

encoder1 = models.encoder0(cascadeLevel = 1 )
albedoDecoder1 = models.decoder0(mode=0 )
normalDecoder1 = models.decoder0(mode=1 )
roughDecoder1 = models.decoder0(mode=2 )
depthDecoder1 = models.decoder0(mode=4 )

lightEncoder = models.encoderLight(cascadeLevel = 0, SGNum = opt.SGNum )
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
        torch.load('{0}/encoder{1}_{2}.pth'.format(opt.experimentBRDF0, 0,
            opt.nepochBRDF0-1 ) ).state_dict() )
albedoDecoder.load_state_dict(
        torch.load('{0}/albedo{1}_{2}.pth'.format(opt.experimentBRDF0, 0,
            opt.nepochBRDF0-1 ) ).state_dict() )
normalDecoder.load_state_dict(
        torch.load('{0}/normal{1}_{2}.pth'.format(opt.experimentBRDF0, 0,
            opt.nepochBRDF0-1 ) ).state_dict() )
roughDecoder.load_state_dict(
        torch.load('{0}/rough{1}_{2}.pth'.format(opt.experimentBRDF0, 0,
            opt.nepochBRDF0-1 ) ).state_dict() )
depthDecoder.load_state_dict(
        torch.load('{0}/depth{1}_{2}.pth'.format(opt.experimentBRDF0, 0,
            opt.nepochBRDF0-1 ) ).state_dict() )
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

lightEncoder.load_state_dict(
        torch.load('{0}/lightEncoder{1}_{2}.pth'.format(opt.experimentLight0,
            0, opt.nepochLight0-1 ) ).state_dict() )
axisDecoder.load_state_dict(
        torch.load('{0}/axisDecoder{1}_{2}.pth'.format(opt.experimentLight0,
            0, opt.nepochLight0-1 ) ).state_dict() )
lambDecoder.load_state_dict(
        torch.load('{0}/lambDecoder{1}_{2}.pth'.format(opt.experimentLight0,
            0, opt.nepochLight0-1 ) ).state_dict() )
weightDecoder.load_state_dict(
        torch.load('{0}/weightDecoder{1}_{2}.pth'.format(opt.experimentLight0,
            0, opt.nepochLight0-1 ) ).state_dict() )
for param in lightEncoder.parameters():
    param.requires_grad = False
for param in axisDecoder.parameters():
    param.requires_grad = False
for param in lambDecoder.parameters():
    param.requires_grad = False
for param in weightDecoder.parameters():
    param.requires_grad = False

if opt.isFineTune:
    encoder1.load_state_dict(
            torch.load('{0}/encoder{1}_{2}.pth'.format(opt.experiment,
                1, opt.nepochFineTune-1) ).state_dict() )
    albedoDecoder1.load_state_dict(
            torch.load('{0}/albedo{1}_{2}.pth'.format(opt.experiment,
                1, opt.nepochFineTune-1) ).state_dict() )
    normalDecoder1.load_state_dict(
            torch.load('{0}/normal{1}_{2}.pth'.format(opt.experiment,
                1, opt.nepochFineTune-1) ).state_dict() )
    roughDecoder1.load_state_dict(
            torch.load('{0}/rough{1}_{2}.pth'.format(opt.experiment,
                1, opt.nepochFineTune-1) ).state_dict() )
    depthDecoder1.load_state_dict(
            torch.load('{0}/depth{1}_{2}.pth'.format(opt.experiment,
                1, opt.nepochFineTune-1) ).state_dict() )
else:
    encoder1.load_state_dict(
            torch.load('{0}/encoder{1}_{2}.pth'.format(opt.experimentBRDF1,
                1, opt.nepochBRDF1-1) ).state_dict() )
    albedoDecoder1.load_state_dict(
            torch.load('{0}/albedo{1}_{2}.pth'.format(opt.experimentBRDF1,
                1, opt.nepochBRDF1-1) ).state_dict() )
    normalDecoder1.load_state_dict(
            torch.load('{0}/normal{1}_{2}.pth'.format(opt.experimentBRDF1,
                1, opt.nepochBRDF1-1) ).state_dict() )
    roughDecoder1.load_state_dict(
            torch.load('{0}/rough{1}_{2}.pth'.format(opt.experimentBRDF1,
                1, opt.nepochBRDF1-1) ).state_dict() )
    depthDecoder1.load_state_dict(
            torch.load('{0}/depth{1}_{2}.pth'.format(opt.experimentBRDF1,
                1, opt.nepochBRDF1-1) ).state_dict() )

lr_scale = 1
#########################################
encoder = nn.DataParallel(encoder, device_ids = opt.deviceIds )
albedoDecoder = nn.DataParallel(albedoDecoder, device_ids = opt.deviceIds )
normalDecoder = nn.DataParallel(normalDecoder, device_ids = opt.deviceIds )
roughDecoder = nn.DataParallel(roughDecoder, device_ids = opt.deviceIds )
depthDecoder = nn.DataParallel(depthDecoder, device_ids = opt.deviceIds )

encoder1 = nn.DataParallel(encoder1, device_ids = opt.deviceIds )
albedoDecoder1 = nn.DataParallel(albedoDecoder1, device_ids = opt.deviceIds )
normalDecoder1 = nn.DataParallel(normalDecoder1, device_ids = opt.deviceIds )
roughDecoder1 = nn.DataParallel(roughDecoder1, device_ids = opt.deviceIds )
depthDecoder1 = nn.DataParallel(depthDecoder1, device_ids = opt.deviceIds )

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

    encoder1 = encoder1.cuda(opt.gpuId )
    albedoDecoder1 = albedoDecoder1.cuda(opt.gpuId )
    normalDecoder1 = normalDecoder1.cuda(opt.gpuId )
    roughDecoder1 = roughDecoder1.cuda(opt.gpuId )
    depthDecoder1 = depthDecoder1.cuda(opt.gpuId )

    lightEncoder = lightEncoder.cuda(opt.gpuId )
    axisDecoder = axisDecoder.cuda(opt.gpuId )
    lambDecoder = lambDecoder.cuda(opt.gpuId )
    weightDecoder = weightDecoder.cuda(opt.gpuId )
####################################


####################################
# Optimizer
opEncoder = optim.Adam(encoder1.parameters(), lr=1e-4 * lr_scale, betas=(0.5, 0.999) )
opAlbedo = optim.Adam(albedoDecoder1.parameters(), lr=1e-4 * lr_scale, betas=(0.5, 0.999) )
opNormal = optim.Adam(normalDecoder1.parameters(), lr=1e-4 * lr_scale, betas=(0.5, 0.999) )
opRough = optim.Adam(roughDecoder1.parameters(), lr=1e-4 * lr_scale, betas=(0.5, 0.999) )
opDepth = optim.Adam(depthDecoder1.parameters(), lr=1e-4 * lr_scale, betas=(0.5, 0.999) )
#####################################


####################################
brdfDataset = dataLoader_ours.BatchLoader( opt.dataRoot,
        imWidth = opt.imWidth1, imHeight = opt.imHeight1,
        cascadeLevel = 0, isLight = False, phase = 'TRAIN' )
IIWDataset = dataLoader_iiw.IIWLoader(
        dataRoot = opt.IIWRoot,
        imHeight = opt.imHeight1,
        imWidth = opt.imWidth1,
        phase = 'TRAIN' )
trainDataset = dataLoader_iiw.ConcatDataset(brdfDataset, IIWDataset )
brdfLoader = DataLoader(trainDataset, batch_size = opt.batchSize,
        num_workers = 8, shuffle = False )

j = 0
# BRDFLost
albedoErrsNpList = np.ones( [1, 1], dtype = np.float32 )
normalErrsNpList = np.ones( [1, 1], dtype = np.float32 )
roughErrsNpList= np.ones( [1, 1], dtype = np.float32 )
depthErrsNpList = np.ones( [1, 1], dtype = np.float32 )

eqErrsNpList = np.ones([1, 1], dtype=np.float32 )
darkerErrsNpList = np.ones([1, 1], dtype=np.float32 )

for epoch in list(range(opt.nepochFineTune, opt.nepoch) ):
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

        ###############################################################################
        ##################### Repetitive Preprocessing ################################
        # Load the image from cpu to gpu
        im_cpu = (dataBatch['im'] )
        imBatch = Variable(im_cpu ).cuda()
        imBatch = F.adaptive_avg_pool2d(imBatch, (opt.imHeight0, opt.imWidth0) )

        x1, x2, x3, x4, x5, x6 = encoder(imBatch )
        albedoPred = 0.5 * (albedoDecoder(imBatch, x1, x2, x3, x4, x5, x6) + 1)
        normalPred = normalDecoder(imBatch, x1, x2, x3, x4, x5, x6)
        roughPred = roughDecoder(imBatch, x1, x2, x3, x4, x5, x6)
        depthPred = 0.5 * (depthDecoder(imBatch, x1, x2, x3, x4, x5, x6) + 1)

        # Normalize Albedo and depth
        bn, ch, nrow, ncol = albedoPred.size()
        albedoPred = albedoPred.view(bn, -1)
        albedoPred = albedoPred / torch.clamp(torch.mean(albedoPred, dim=1), min=1e-10).unsqueeze(1) / 3.0
        albedoPred = albedoPred.view(bn, ch, nrow, ncol)

        bn, ch, nrow, ncol = depthPred.size()
        depthPred = depthPred.view(bn, -1)
        depthPred = depthPred / torch.clamp(torch.mean(depthPred, dim=1), min=1e-10).unsqueeze(1) / 3.0
        depthPred = depthPred.view(bn, ch, nrow, ncol )

        imBatchLarge = F.interpolate(imBatch, [480, 640], mode='bilinear')
        albedoPredLarge = F.interpolate(albedoPred, [480, 640], mode='bilinear')
        normalPredLarge = F.interpolate(normalPred, [480, 640], mode='bilinear')
        roughPredLarge = F.interpolate(roughPred, [480,640], mode='bilinear')
        depthPredLarge = F.interpolate(depthPred, [480, 640], mode='bilinear')

        inputBatch = torch.cat([imBatchLarge, albedoPredLarge,
            0.5*(normalPredLarge+1), 0.5*(roughPredLarge+1), depthPredLarge ], dim=1 )
        x1, x2, x3, x4, x5, x6 = lightEncoder(inputBatch )

        # Prediction
        imBatchSmall = F.adaptive_avg_pool2d(imBatch, (opt.envRow, opt.envCol) )
        axisPred = axisDecoder(x1, x2, x3, x4, x5, x6, imBatchSmall )
        lambPred = lambDecoder(x1, x2, x3, x4, x5, x6,imBatchSmall )
        weightPred = weightDecoder(x1, x2, x3, x4, x5, x6, imBatchSmall )
        bn, SGNum, _, envRow, envCol = axisPred.size()
        envmapsPred = torch.cat([axisPred.view(bn, SGNum*3, envRow, envCol ), lambPred, weightPred], dim=1)

        envmapsPredImage, axisPred, lambPred, weightPred = output2env.output2env(axisPred, lambPred, weightPred )

        envmapsPredImage.requires_grad = False
        diffusePred, specularPred = renderLayer.forwardEnv(albedoPred, normalPred,
                roughPred, envmapsPredImage )

        diffusePred, specularPred = models.LSregressDiffSpec(
                diffusePred,
                specularPred,
                imBatchSmall,
                diffusePred, specularPred )

        dataBatch['albedoPre'] = albedoPred.detach()
        dataBatch['normalPre'] = 0.5 * (normalPred.detach()+1)
        dataBatch['roughPre'] = 0.5 * (roughPred.detach() + 1)
        dataBatch['depthPre'] = depthPred.detach()
        dataBatch['diffusePre'] = diffusePred.detach()
        dataBatch['specularPre'] = specularPred.detach()
        dataBatch['envmapsPre'] = envmapsPred.detach()
        ##################################################################################################

        albedoPair, normalPair, roughPair, depthPair,  \
        = wcg.wrapperBRDF(dataBatch, opt, encoder1, \
        albedoDecoder1, normalDecoder1, roughDecoder1, depthDecoder1 )

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
        utils.writeErrToScreen('albedo', [albedoErr], epoch, j )
        utils.writeErrToScreen('normal', [normalErr], epoch, j )
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

        ###############################################################################
        ##################### Repetitive Preprocessing ################################
        # Load the image from cpu to gpu
        im_cpu = (IIWBatch['im'] )
        imBatch = Variable(imBatch ).cuda()
        imBatch = F.adaptive_avg_pool2d(imBatch, (opt.imHeight0, opt.imWidth0) )

        x1, x2, x3, x4, x5, x6 = encoder(imBatch )
        albedoPred = 0.5 * (albedoDecoder(imBatch, x1, x2, x3, x4, x5, x6) + 1)
        normalPred = normalDecoder(imBatch, x1, x2, x3, x4, x5, x6)
        roughPred = roughDecoder(imBatch, x1, x2, x3, x4, x5, x6)
        depthPred = 0.5 * (depthDecoder(imBatch, x1, x2, x3, x4, x5, x6) + 1)

        # Normalize Albedo and depth
        bn, ch, nrow, ncol = albedoPred.size()
        albedoPred = albedoPred.view(bn, -1)
        albedoPred = albedoPred / torch.clamp(torch.mean(albedoPred, dim=1), min=1e-10).unsqueeze(1) / 3.0
        albedoPred = albedoPred.view(bn, ch, nrow, ncol)

        bn, ch, nrow, ncol = depthPred.size()
        depthPred = depthPred.view(bn, -1)
        depthPred = depthPred / torch.clamp(torch.mean(depthPred, dim=1), min=1e-10).unsqueeze(1) / 3.0
        depthPred = depthPred.view(bn, ch, nrow, ncol)

        imBatchLarge = F.interpolate(imBatch, [480, 640], mode='bilinear')
        albedoPredLarge = F.interpolate(albedoPred, [480, 640], mode='bilinear')
        normalPredLarge = F.interpolate(normalPred, [480, 640], mode='bilinear')
        roughPredLarge = F.interpolate(roughPred, [480,640], mode='bilinear')
        depthPredLarge = F.interpolate(depthPred, [480, 640], mode='bilinear')

        inputBatch = torch.cat([imBatchLarge, albedoPredLarge,
            0.5*(normalPredLarge+1), 0.5*(roughPredLarge+1), depthPredLarge ], dim=1 )
        x1, x2, x3, x4, x5, x6 = lightEncoder(inputBatch )

        # Prediction
        imBatchSmall = F.adaptive_avg_pool2d(imBatch, (opt.envRow, opt.envCol) )
        axisPred = axisDecoder(x1, x2, x3, x4, x5, x6, imBatchSmall )
        lambPred = lambDecoder(x1, x2, x3, x4, x5, x6, imBatchSmall )
        weightPred = weightDecoder(x1, x2, x3, x4, x5, x6, imBatchSmall )
        bn, SGNum, _, envRow, envCol = axisPred.size()
        envmapsPred = torch.cat([axisPred.view(bn, SGNum*3, envRow, envCol ), lambPred, weightPred], dim=1)

        envmapsPredImage, axisPred, lambPred, weightPred = output2env.output2env(axisPred, lambPred, weightPred )

        diffusePred, specularPred = renderLayer.forwardEnv(albedoPred, normalPred,
                roughPred, envmapsPredImage )

        diffusePred, specularPred = models.LSregressDiffSpec(
                diffusePred,
                specularPred,
                imBatchSmall,
                diffusePred, specularPred )

        IIWBatch['albedoPre'] = albedoPred.detach()
        IIWBatch['normalPre'] = 0.5 * (normalPred.detach() + 1)
        IIWBatch['roughPre'] = 0.5 * (roughPred.detach() + 1)
        IIWBatch['depthPre'] = depthPred.detach()
        IIWBatch['diffusePre'] = diffusePred.detach()
        IIWBatch['specularPre'] = specularPred.detach()
        IIWBatch['envmapsPre'] = envmapsPred.detach()
        ##################################################################################################

        albedoPair, normalPair, roughPair, depthPair,  \
        eqPair, darkerPair \
        = wiiw.wrapperIIW(IIWBatch, opt, encoder1, \
        albedoDecoder1, normalDecoder1, roughDecoder1, depthDecoder1 )

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
            vutils.save_image( ( (imBatch)**(1.0/2.2) ).data,
                    '{0}/{1}_imIIW.png'.format(opt.experiment, j) )

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

        if j % 1000 == 0:
            # save the models
            torch.save(encoder1.module, '{0}/encoder{1}_{2}_{3}.pth'.format(opt.experiment, opt.cascadeLevel, epoch, j ) )
            torch.save(albedoDecoder1.module, '{0}/albedo{1}_{2}_{3}.pth'.format(opt.experiment, opt.cascadeLevel, epoch, j ) )
            torch.save(normalDecoder1.module, '{0}/normal{1}_{2}_{3}.pth'.format(opt.experiment, opt.cascadeLevel, epoch, j ) )
            torch.save(roughDecoder1.module, '{0}/rough{1}_{2}_{3}.pth'.format(opt.experiment, opt.cascadeLevel, epoch, j ) )
            torch.save(depthDecoder1.module, '{0}/depth{1}_{2}_{3}.pth'.format(opt.experiment, opt.cascadeLevel, epoch, j ) )
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
    torch.save(encoder1.module, '{0}/encoder{1}_{2}.pth'.format(opt.experiment, opt.cascadeLevel, epoch) )
    torch.save(albedoDecoder1.module, '{0}/albedo{1}_{2}.pth'.format(opt.experiment, opt.cascadeLevel, epoch) )
    torch.save(normalDecoder1.module, '{0}/normal{1}_{2}.pth'.format(opt.experiment, opt.cascadeLevel, epoch) )
    torch.save(roughDecoder1.module, '{0}/rough{1}_{2}.pth'.format(opt.experiment, opt.cascadeLevel, epoch) )
    torch.save(depthDecoder1.module, '{0}/depth{1}_{2}.pth'.format(opt.experiment, opt.cascadeLevel, epoch) )
