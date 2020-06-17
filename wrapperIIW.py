import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import models

# Return triplet of predictions, ground-truth and error
def wrapperIIW(dataBatch, opt,
    encoder, albedoDecoder, normalDecoder, roughDecoder, depthDecoder ):


    eq = dataBatch['eq']
    darker = dataBatch['darker']

    # Load the image from cpu to gpu
    im_cpu = dataBatch['im']
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
                diffusePreBatch.detach(),
                specularPreBatch.detach(),
                imBatchSmall,
                diffusePreBatch, specularPreBatch )

        if diffusePreBatch.size(2) < opt.imHeight or diffusePreBatch.size(3) < opt.imWidth:
            diffusePreBatch = F.interpolate(diffusePreBatch, [opt.imHeight, opt.imWidth ], mode='bilinear')
        if specularPreBatch.size(2) < opt.imHeight or specularPreBatch.size(3) < opt.imWidth:
            specularPreBatch = F.interpolate(specularPreBatch, [opt.imHeight, opt.imWidth ], mode='bilinear')

        # Normalize Albedo and depth
        bn, ch, nrow, ncol = albedoPreBatch.size()
        albedoPreBatch = albedoPreBatch.view(bn, -1)
        albedoPreBatch = albedoPreBatch / torch.clamp(torch.mean(albedoPreBatch, dim=1), min=1e-10).unsqueeze(1) / 3.0
        albedoPreBatch = albedoPreBatch.view(bn, ch, nrow, ncol)

        bn, ch, nrow, ncol = depthPreBatch.size()
        depthPreBatch = depthPreBatch.view(bn, -1)
        depthPreBatch = depthPreBatch / torch.clamp(torch.mean(depthPreBatch, dim=1), min=1e-10).unsqueeze(1) / 3.0
        depthPreBatch = depthPreBatch.view(bn, ch, nrow, ncol)

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

    eqLoss, darkerLoss = 0, 0
    for m in range(0, albedoPred.size(0) ):
        eqPoint = eq['point'][m, :].numpy().astype(np.long )
        eqWeight = eq['weight'][m, :].numpy().astype(np.float32 )
        eqNum = eq['num'][m].numpy().astype(np.long )
        eqPoint = eqPoint[0:eqNum, :]
        eqWeight = eqWeight[0:eqNum ]

        darkerPoint = darker['point'][m, :].numpy().astype(np.long )
        darkerWeight = darker['weight'][m, :].numpy().astype(np.float32 )
        darkerNum = darker['num'][m].numpy().astype(np.long )
        darkerPoint = darkerPoint[0:darkerNum, :]
        darkerWeight = darkerWeight[0:darkerNum ]
        eqL, darkerL = \
                models.BatchRankingLoss(albedoPred[m, :],
                        eqPoint, eqWeight,
                        darkerPoint, darkerWeight )
        eqLoss += eqL
        darkerLoss += darkerL

    eqLoss = eqLoss / max(albedoPred.size(0 ), 1e-5)
    darkerLoss = darkerLoss / max(albedoPred.size(0), 1e-5)

    return [albedoPred, None], [normalPred, None], \
            [roughPred, None], [depthPred, None], \
            [eq, eqLoss], [darker, darkerLoss]

