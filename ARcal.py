#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from joblib import Parallel, delayed
from moco.GRU import *
from feeder.augmentations import  *
from rpy2.robjects import r
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import rpy2.robjects as robjects

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from dataset import get_finetune_validation_set

parser = argparse.ArgumentParser(description='TVAR')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=80, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')


parser.add_argument('--finetune-dataset', default='ntu60', type=str,
                    help='which dataset to use for finetuning')

parser.add_argument('--protocol', default='cross_view', type=str,
                    help='traiining protocol of ntu')
parser.add_argument('--finetune-skeleton-representation', default='graph-based', type=str,
                    help='which skeleton-representation to use for downstream training')



args = parser.parse_args()

def rcal1(inputnp, num_of_frame, label, c, r, d, e):
    tvrge = importr("tvReg")  # library R package
    pandas2ri.activate()
    Num = 300 // num_of_frame[r] + 1
    ## First body
    for i in range(25):
        for k in range(3):
            inputnp1 = inputnp[r, k, :num_of_frame[r], i, 0]
            inputnp1 = inputnp1.tolist()
            inputnp1 = robjects.FloatVector(inputnp1)
            # AR2
            result = tvrge.tvAR(inputnp1, p=2, type="none", est="ll")
            coef = result.rx2("coefficients").T
            coef = np.squeeze(coef)
            coef0 = coef[0, :]

            coef0 = np.pad(coef0, (0, 2), 'constant', constant_values=(0, 0))
            coef0 = np.tile(coef0, Num)
            coef0 = coef0[:300]
            coef1 = coef[1, :]
            coef1 = np.pad(coef1, (0, 2), 'constant', constant_values=(0, 0))
            coef1 = np.tile(coef1, Num)
            coef1 = coef1[:300]
            # AR1
            result1 = tvrge.tvAR(inputnp1, p=1, type="none", est="ll")
            coefar1 = result1.rx2("coefficients").T
            coefar1 = np.squeeze(coefar1)
            coefar1 = np.pad(coefar1, (0, 1), 'constant', constant_values=(0, 0))
            coefar1 = np.tile(coefar1, Num)
            coefar1 = coefar1[:300]
            # replace the first parameters
            coef01 = coef[0, :]
            coef01 = np.pad(coef01, (1, 1), 'constant', constant_values=(0, 0))
            coef01[0] = coefar1[0]
            coef01 = np.tile(coef01, Num)
            coef01 = coef01[:300]

            c = c.copy()
            c[k, :, i, 0] = coef0
            d = d.copy()
            d[k, :, i, 0] = coef1
            e = e.copy()
            e[k, :, i, 0] = coef01
    ## Second body
    for i in range(25):
        for k in range(3):
            inputnp2 = inputnp[r, k, :num_of_frame[r], i, 1]
            inputnp2 = inputnp2.tolist()
            inputnp2 = robjects.FloatVector(inputnp2)
            result = tvrge.tvAR(inputnp2, p=2, type="none", est="ll")

            coef = result.rx2("coefficients").T
            coef = np.squeeze(coef)
            coef0 = coef[0, :]
            coef0 = np.pad(coef0, (0, 2), 'constant', constant_values=(0, 0))
            coef0 = np.tile(coef0, Num)
            coef0 = coef0[:300]
            coef1 = coef[1, :]
            coef1 = np.pad(coef1, (0, 2), 'constant', constant_values=(0, 0))
            coef1 = np.tile(coef1, Num)
            coef1 = coef1[:300]

            result1 = tvrge.tvAR(inputnp2, p=1, type="none", est="ll")
            coefar2 = result1.rx2("coefficients").T
            coefar2 = np.squeeze(coefar2)
            coefar2 = np.pad(coefar2, (0, 1), 'constant', constant_values=(0, 0))
            coefar2 = np.tile(coefar2, Num)
            coefar2 = coefar2[:300]

            coef02 = coef[0, :]
            coef02 = np.pad(coef02, (1, 1), 'constant', constant_values=(0, 0))
            coef02[0] = coefar2[0]
            coef02 = np.tile(coef02, Num)
            coef02 = coef02[:300]


            c[k, :, i, 1] = coef0
            d[k, :, i, 1] = coef1
            e[k, :, i, 1] = coef02
    return c, d, e

def autore(input, num_of_frame, label ):
    inputnp = input.detach().cpu().numpy()
    num_of_frame = num_of_frame.detach().cpu().numpy()
    label =label.detach().cpu().numpy()
    c = np.zeros((3, 300, 25, 2))
    d = np.zeros((3, 300, 25, 2))
    e = np.zeros((3, 300, 25, 2))
    # Parallel
    results= Parallel(n_jobs=8)(
        delayed(rcal1)(inputnp, num_of_frame, label, c, r, d, e)
        for r in range(8))
    results = np.asarray(results)
    c = results[:, 0, :, :, :, :].squeeze()
    d = results[:, 1, :, :, :, :].squeeze()
    e = results[:, 2, :, :, :, :].squeeze()
    c = e*c+d
    c[np.isnan(c)] = 0
    e[np.isnan(e)] = 0
    return c, e
def  ARcalculate( data_eval):
    for ith, (ith_data, label, number_of_frame) in enumerate(data_eval):
        input_tensor = ith_data.to(device)
        input = input_tensor.clone()

        vel, vel1 = autore(input, number_of_frame, label)
        np.savez_compressed('../../ARmodel/NTU%dCVval.npz' % (ith), AR2=vel, AR1=vel1)

    return input




def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True


    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count()
    # Simply call main_worker function
    main_worker(0, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    from options import options_attack as options
    if args.finetune_dataset == 'ntu60' and args.protocol == 'cross_view':
        opts = options.opts_ntu_60_cross_view()
    elif args.finetune_dataset == 'ntu60' and args.protocol == 'cross_subject':
        opts = options.opts_ntu_60_cross_subject()

    opts.test_feeder_args['input_representation'] = args.finetune_skeleton_representation

    cudnn.benchmark = True

    ## Data loading code

    val_dataset = get_finetune_validation_set(opts)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    results = ARcalculate(val_loader)



if __name__ == '__main__':
    main()
