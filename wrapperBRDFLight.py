import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import models

# Return triplet of predictions, ground-truth and error
def wrapperBRDFLight(dataBatch, opt,
    encoder, albedoDecoder, normalDecoder, roughDecoder, depthDecoder,
    lightEncoder, axisDecoder, lambDecoder, weightDecoder,
    output2env, renderLayer, offset = 1.0, isLightOut = False ):

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
    seg_cpu = torch.cat( [segArea_cpu, segEnv_cpu, segObj_cpu], dim=1 )
    segBatch = Variable( seg_cpu ).cuda()

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
                diffusePreBatch.detach(), specularPreBatch.detach(),
                imBatchSmall,
                diffusePreBatch, specularPreBatch )

        if diffusePreBatch.size(2) < opt.imHeight or diffusePreBatch.size(3) < opt.imWidth:
            diffusePreBatch = F.interpolate(diffusePreBatch, [opt.imHeight, opt.imWidth ], mode='bilinear')
        if specularPreBatch.size(2) < opt.imHeight or specularPreBatch.size(3) < opt.imWidth:
            specularPreBatch = F.interpolate(specularPreBatch, [opt.imHeight, opt.imWidth ], mode='bilinear')

        envmapsPre_cpu = dataBatch['envmapsPre']
        envmapsPreBatch = Variable(envmapsPre_cpu ).cuda()

        # Normalize Albedo and depth
        bn, ch, nrow, ncol = albedoPreBatch.size()
        albedoPreBatch = albedoPreBatch.view(bn, -1)
        albedoPreBatch = albedoPreBatch / torch.clamp(torch.mean(albedoPreBatch, dim=1), min=1e-10).unsqueeze(1) / 3.0
        albedoPreBatch = albedoPreBatch.view(bn, ch, nrow, ncol)

        bn, ch, nrow, ncol = depthPreBatch.size()
        depthPreBatch = depthPreBatch.view(bn, -1)
        depthPreBatch = depthPreBatch / torch.clamp(torch.mean(depthPreBatch, dim=1), min=1e-10).unsqueeze(1) / 3.0
        depthPreBatch = depthPreBatch.view(bn, ch, nrow, ncol)


    envmaps_cpu = dataBatch['envmaps']
    envmapsBatch = Variable(envmaps_cpu ).cuda()

    envmapsInd_cpu = dataBatch['envmapsInd']
    envmapsIndBatch = Variable(envmapsInd_cpu ).cuda()

    ########################################################
    # Build the cascade network architecture #
    if opt.cascadeLevel == 0:
        inputBatch = imBatch
    elif opt.cascadeLevel > 0:
        inputBatch = torch.cat([imBatch, albedoPreBatch,
            normalPreBatch, roughPreBatch, depthPreBatch,
            diffusePreBatch, specularPreBatch ], dim=1)

    # Initial Prediction
    x1, x2, x3, x4, x5, x6 = encoder(inputBatch )
    albedoPred = 0.5 * (albedoDecoder(imBatch, x1, x2, x3, x4, x5, x6) + 1)
    normalPred = normalDecoder(imBatch, x1, x2, x3, x4, x5, x6)
    roughPred = roughDecoder(imBatch, x1, x2, x3, x4, x5, x6)
    depthPred = 0.5 * (depthDecoder(imBatch, x1, x2, x3, x4, x5, x6) + 1)

    albedoBatch = segBRDFBatch * albedoBatch
    albedoPred1 = models.LSregress(albedoPred.detach() * segBRDFBatch.expand_as(albedoPred),
            albedoBatch * segBRDFBatch.expand_as(albedoBatch), albedoPred ) 
    albedoPred1 = torch.clamp(albedoPred1, 0, 1)

    depthPred1 = models.LSregress(depthPred.detach() *  segAllBatch.expand_as(depthPred),
            depthBatch * segAllBatch.expand_as(depthBatch), depthPred)

    ## Compute Errors
    pixelObjNum = (torch.sum(segBRDFBatch ).cpu().data).item()
    pixelAllNum = (torch.sum(segAllBatch ).cpu().data).item()

    albedoErr = torch.sum( (albedoPred1 - albedoBatch )
            * (albedoPred1 - albedoBatch) * segBRDFBatch.expand_as(albedoBatch) / pixelObjNum / 3.0)
    normalErr = torch.sum( (normalPred - normalBatch)
        * (normalPred - normalBatch) * segAllBatch.expand_as(normalBatch) ) / pixelAllNum / 3.0
    roughErr = torch.sum( (roughPred - roughBatch)
        * (roughPred - roughBatch) * segBRDFBatch ) / pixelObjNum
    depthErr = torch.sum( (torch.log(depthPred1 + 1) - torch.log(depthBatch + 1) )
            * ( torch.log(depthPred1 + 1) - torch.log(depthBatch + 1) ) * segAllBatch.expand_as(depthBatch ) ) / pixelAllNum

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
        0.5*(normalPredLarge+1), 0.5 * (roughPredLarge+1), depthPredLarge ], dim=1 )

    if opt.cascadeLevel == 0:
        x1, x2, x3, x4, x5, x6 = lightEncoder(inputBatch.detach() )
    elif opt.cascadeLevel > 0:
        x1, x2, x3, x4, x5, x6 = lightEncoder(inputBatch.detach(), envmapsPreBatch.detach() )

    # Prediction
    axisPred = axisDecoder(x1, x2, x3, x4, x5, x6, envmapsBatch )
    lambPred = lambDecoder(x1, x2, x3, x4, x5, x6, envmapsBatch )
    weightPred = weightDecoder(x1, x2, x3, x4, x5, x6, envmapsBatch )
    bn, SGNum, _, envRow, envCol = axisPred.size()
    envmapsPred = torch.cat([axisPred.view(bn, SGNum * 3, envRow, envCol ), lambPred, weightPred], dim=1)

    imBatchSmall = F.adaptive_avg_pool2d(imBatch, (opt.envRow, opt.envCol) )
    segBatchSmall = F.adaptive_avg_pool2d(segBRDFBatch, (opt.envRow, opt.envCol) )
    notDarkEnv = (torch.mean(torch.mean(torch.mean(envmapsBatch, 4), 4), 1, True ) > 0.001 ).float()
    segEnvBatch = (segBatchSmall * envmapsIndBatch.expand_as(segBatchSmall) ).unsqueeze(-1).unsqueeze(-1)
    segEnvBatch = segEnvBatch * notDarkEnv.unsqueeze(-1).unsqueeze(-1)
    
    # Compute the recontructed error
    envmapsPredImage, axisPred, lambPred, weightPred = output2env.output2env(axisPred, lambPred, weightPred )

    pixelNum = max( (torch.sum(segEnvBatch ).cpu().data).item(), 1e-5)
    envmapsPredScaledImage = models.LSregress(envmapsPredImage.detach() * segEnvBatch.expand_as(envmapsBatch ),
            envmapsBatch * segEnvBatch.expand_as(envmapsBatch), envmapsPredImage )

    reconstErr = torch.sum( ( torch.log(envmapsPredScaledImage + offset ) -
        torch.log(envmapsBatch + offset ) )
        * ( torch.log(envmapsPredScaledImage + offset ) -
            torch.log(envmapsBatch + offset ) ) *
        segEnvBatch.expand_as(envmapsPredImage ) ) \
        / pixelNum / 3.0 / opt.envWidth / opt.envHeight


    # Compute the rendered error
    pixelNum = max( (torch.sum(segBatchSmall ).cpu().data).item(), 1e-5 )

    diffusePred, specularPred = renderLayer.forwardEnv(albedoPred.detach(), normalPred,
            roughPred, envmapsPredImage )

    diffusePredScaled, specularPredScaled = models.LSregressDiffSpec(
            diffusePred.detach(),
            specularPred.detach(),
            imBatchSmall,
            diffusePred, specularPred )

    renderedImPred = torch.clamp(diffusePredScaled + specularPredScaled, 0, 1)

    renderErr = torch.sum( (renderedImPred - imBatchSmall)
        * (renderedImPred - imBatchSmall) * segBatchSmall.expand_as(imBatchSmall )  ) \
        / pixelNum / 3.0

    if isLightOut == False:
        return [albedoPred, albedoErr, albedoBatch], \
                [normalPred, normalErr, normalBatch ], \
                [roughPred, roughErr, roughBatch], \
                [depthPred, depthErr, depthBatch], \
                [envmapsPredScaledImage, reconstErr, envmapsBatch], \
                [renderedImPred, renderErr, imBatch] 
    else:
        return [albedoPred, albedoErr, albedoBatch], \
                [normalPred, normalErr, normalBatch], \
                [roughPred, roughErr, roughBatch ], \
                [depthPred, depthErr, depthBatch], \
                [envmapsPredScaledImage, reconstErr], \
                [renderedImPred, renderErr, imBatch], \
                [envmapsPred, diffusePred, specularPred]
