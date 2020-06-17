import torch
import argparse
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import sys

import BilateralGrid as bs


##############################################################################
REQUIRES_CONF_GRAD = True
##############################################################################

class BilateralFunction(torch.autograd.Function):


    @staticmethod
    def forward(ctx, image, pred, confidence, grid_params_arr, bs_params_arr ):

        batch_size, channel_num, height, width = pred.size()

        output = np.zeros((batch_size, height, width, channel_num), np.float32)
        yhat_list = []

        image_np = image.cpu().numpy().swapaxes(1, 2).swapaxes(2, 3)
        pred_np = pred.cpu().numpy().swapaxes(1, 2).swapaxes(2, 3)
        conf_np = confidence.cpu().numpy().squeeze(1)

        grid_params = {}
        grid_params['sigma_luma'] = grid_params_arr[0].data.item()
        grid_params['sigma_chroma'] = grid_params_arr[1].data.item()
        grid_params['sigma_spatial'] = grid_params_arr[2].data.item()

        bs_params = {}
        bs_params['lam'] =  bs_params_arr[0].data.item()
        bs_params['A_diag_min'] = bs_params_arr[1].data.item()
        bs_params['cg_tol'] = bs_params_arr[2].data.item()
        bs_params['cg_maxiter'] = bs_params_arr[2].data.item()

        for i in range(batch_size):
            curr_image = image_np[i, :, :, :]
            curr_pred = pred_np[i, :, :, :]
            curr_conf = conf_np[i, :, :]
            im_shape = curr_pred.shape

            grid = bs.BilateralGrid(curr_image*255.0, **grid_params)

            curr_result, yhat = bs.solve(grid, curr_pred, curr_conf, bs_params, im_shape)
            output[i, :, :, :] = curr_result
            #  print yhat.shape
            yhat_list.append(yhat)

        ctx.save_for_backward(image, pred, confidence, grid_params_arr, bs_params_arr)
        ctx.intermediate_results = yhat_list

        output = output.swapaxes(3, 2).swapaxes(2, 1)
        return torch.Tensor(output).cuda(), confidence

    @staticmethod
    def backward(ctx, grad_output, grad_not_used ):

        image, pred, confidence, grid_params_arr, bs_params_arr = ctx.saved_variables

        grid_params = {}
        grid_params['sigma_luma'] = grid_params_arr[0].data.item()
        grid_params['sigma_chroma'] = grid_params_arr[1].data.item()
        grid_params['sigma_spatial'] = grid_params_arr[2].data.item()

        bs_params = {}
        bs_params['lam'] =  bs_params_arr[0].data.item()
        bs_params['A_diag_min'] = bs_params_arr[1].data.item()
        bs_params['cg_tol'] = bs_params_arr[2].data.item()
        bs_params['cg_maxiter'] = bs_params_arr[2].data.item()


        yhat_list = ctx.intermediate_results

        batch_size, channel_num, height, width = pred.size()

        # output gradient
        pred_grad = np.zeros((batch_size, height, width, channel_num),
                np.float32)
        conf_grad = np.zeros((batch_size, height, width), np.float32)

        image_np = image.data.cpu().numpy().swapaxes(1, 2).swapaxes(2, 3)
        pred_np = pred.data.cpu().numpy().swapaxes(1, 2).swapaxes(2, 3)
        conf_np = confidence.data.cpu().numpy().squeeze()
        grad_output_np = grad_output.data.cpu().numpy().swapaxes(1, 2).swapaxes(2, 3)

        for i in range(batch_size):
            curr_image = image_np[i, :, :, :]
            curr_grad = grad_output_np[i, :, :, :]
            curr_conf = conf_np[i, :, :]
            curr_yhat = yhat_list[i]
            curr_pred = pred_np[i, :, :, :]

            im_shape = curr_pred.shape
            grid = bs.BilateralGrid(curr_image*255.0, **grid_params)
            curr_pred_grad, curr_conf_grad = bs.solveForGrad(grid,
                    curr_grad, curr_conf, bs_params, im_shape,
                    curr_yhat, curr_pred)

            pred_grad[i, :, :, :] = curr_pred_grad

            if REQUIRES_CONF_GRAD == True:
                conf_grad[i, :, :] = curr_conf_grad

        pred_grad = pred_grad.swapaxes(3, 2).swapaxes(2, 1)
        pred_grad = torch.Tensor(pred_grad).cuda()
        pred_grad = Variable(pred_grad )

        if REQUIRES_CONF_GRAD == True:
            conf_grad = torch.Tensor(conf_grad).cuda().unsqueeze(1)
            conf_grad = Variable(conf_grad)
        else:
            conf_grad = None

        return  None, pred_grad, conf_grad, None, None

class BilateralLayer(nn.Module):
    def __init__(self, mode = 0, isCuda = True, gpuId = 0 ):
        super(BilateralLayer, self).__init__()


        if mode == 0:
            # bilateral solver for albedo
            self.grid_params = {
                'sigma_luma' : 8, #Brightness bandwidth
                'sigma_chroma': 2, # Color bandwidth
                'sigma_spatial': 7# Spatial bandwidth
                }

            self.bs_params = {
                'lam': 200, # The strength of the smoothness parameter
                'A_diag_min': 1e-5, # Clamp the diagonal of the A diagonal in the Jacobi preconditioner.
                'cg_tol': 1e-5, # The tolerance on the convergence in PCG
                'cg_maxiter': 12 # The number of PCG iterations
                }

        elif mode == 1:
            # bilateral solver for normal
            self.grid_params = {
                'sigma_luma' : 0.5, #Brightness bandwidth
                'sigma_chroma': 0.5, # Color bandwidth
                'sigma_spatial': 0.5# Spatial bandwidth
                }

            self.bs_params = {
                'lam': 5, # The strength of the smoothness parameter
                'A_diag_min': 1e-5, # Clamp the diagonal of the A diagonal in the Jacobi preconditioner.
                'cg_tol': 1e-5, # The tolerance on the convergence in PCG
                'cg_maxiter': 10 # The number of PCG iterations
                }

        elif mode == 2:
            # bilateral solver for roughness
            self.grid_params = {
                'sigma_luma' : 8, #Brightness bandwidth
                'sigma_chroma': 2, # Color bandwidth
                'sigma_spatial': 8# Spatial bandwidth
                }

            self.bs_params = {
                'lam': 300, # The strength of the smoothness parameter
                'A_diag_min': 1e-5, # Clamp the diagonal of the A diagonal in the Jacobi preconditioner.
                'cg_tol': 1e-5, # The tolerance on the convergence in PCG
                'cg_maxiter': 10 # The number of PCG iterations
                }

        elif mode == 4:
            # bilateral solver for normal
            self.grid_params = {
                'sigma_luma' : 4, #Brightness bandwidth
                'sigma_chroma': 2, # Color bandwidth
                'sigma_spatial': 4# Spatial bandwidth
                }

            self.bs_params = {
                'lam': 100, # The strength of the smoothness parameter
                'A_diag_min': 1e-5, # Clamp the diagonal of the A diagonal in the Jacobi preconditioner.
                'cg_tol': 1e-5, # The tolerance on the convergence in PCG
                'cg_maxiter': 10 # The number of PCG iterations
                }


        self.grid_params_arr = Variable(torch.FloatTensor(3) )
        self.bs_params_arr = Variable(torch.FloatTensor(4) )

        self.grid_params_arr[0] = self.grid_params['sigma_luma']
        self.grid_params_arr[1] = self.grid_params['sigma_chroma']
        self.grid_params_arr[2] = self.grid_params['sigma_spatial']

        self.bs_params_arr[0] = self.bs_params['lam']
        self.bs_params_arr[1] = self.bs_params['A_diag_min']
        self.bs_params_arr[2] = self.bs_params['cg_tol']
        self.bs_params_arr[3] = self.bs_params['cg_maxiter']

        if isCuda:
            self.grid_params_arr = self.grid_params_arr.cuda(gpuId )
            self.bs_params_arr = self.bs_params_arr.cuda(gpuId )

        self.grid_params_arr.requires_grad = False
        self.bs_params_arr.requires_grad = False

        self.pad1 = nn.ReplicationPad2d(1)
        if mode == 2 or mode == 4:
            self.conv1 = nn.Conv2d(in_channels = 4, out_channels=16, kernel_size=4, stride = 2, bias=True)
        else:
            self.conv1 = nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size=4, stride = 2, bias=True)

        self.gn1 = nn.GroupNorm(num_groups=2, num_channels = 16)

        self.pad2 = nn.ReplicationPad2d(1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=4, stride=2, bias=True)
        self.gn2 = nn.GroupNorm(num_groups=2, num_channels=16)

        self.dconv1 = nn.Conv2d(in_channels=16, out_channels=16,
                kernel_size=3, stride=1, padding = 1, bias=True )
        self.dgn1 = nn.GroupNorm(num_groups=2, num_channels=16 )

        self.dconv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3,
                stride=1, padding = 1, bias=True)
        self.dgn2 = nn.GroupNorm(num_groups=2, num_channels = 16 )

        self.dpad3 = nn.ReplicationPad2d(1)
        self.dconvFinal = nn.Conv2d(in_channels = 16, out_channels = 1, kernel_size = 3, stride=1, bias=True)

    def computePadding(self, os, ns):
        assert(os <= ns )
        gap = ns - os
        if gap % 2 == 0:
            return [int(gap/2), int(gap / 2) ]
        else:
            return [int((gap+1) / 2), int((gap-1) / 2) ]

    def forward(self, image, feature, pred ):

        scale, _ = torch.max(image, dim=1, keepdim = True)
        scale, _ = torch.max(scale, dim=2, keepdim = True)
        scale, _ = torch.max(scale, dim=3, keepdim = True)
        scale = torch.clamp(scale, 1e-5, 1)
        image = image / scale.expand_as(image )

        scale, _ = torch.max(feature, dim=1, keepdim = True)
        scale, _ = torch.max(scale, dim=2, keepdim = True)
        scale, _ = torch.max(scale, dim=3, keepdim = True)
        scale = torch.clamp(scale, 1e-5, 1)
        feature = feature / scale.expand_as(image )

        inputBatch = torch.cat([image, pred ], dim=1).detach()

        x1 = F.relu(self.gn1(self.conv1(self.pad1(inputBatch) ) ), True)
        x2 = F.relu(self.gn2(self.conv2(self.pad2(x1) ) ), True)

        dx1 = F.relu(self.dgn1(self.dconv1(x2 ) ), True)

        dx1 = F.interpolate(dx1, [x1.size(2), x1.size(3)], mode='bilinear')
        xin2 = torch.cat([dx1, x1], dim=1)
        dx2 = F.relu(self.dgn2(self.dconv2(xin2 ) ), True) 

        dx2 = F.interpolate(dx2, [inputBatch.size(2), inputBatch.size(3)],
                mode='bilinear' )
        conf = 0.5* (torch.tanh(self.dconvFinal(self.dpad3(dx2) ) ) + 1 )
        conf = conf / torch.clamp(conf.max(), min = 1e-5)

        return BilateralFunction.apply(feature.detach(), pred, conf, self.grid_params_arr, self.bs_params_arr )


