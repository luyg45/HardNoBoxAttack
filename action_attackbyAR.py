#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import os
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from sklearn import preprocessing
from sklearn.cluster import KMeans
from moco.GRU import *
from moco.AGCN import Model as AGCN
from feeder.augmentations import *



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# change for attack
from dataset import get_finetune_validation_set

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=80, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N',
                    help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=30., type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[50, 70, ], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by a ratio)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0., type=float,
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')

parser.add_argument('--pretrained', default='./checkpoints/checkpoint_0450.pth.tar', type=str,
                    help='path to moco pretrained checkpoint')
parser.add_argument('--finetune-dataset', default='ntu60', type=str,
                    help='which dataset to use for finetuning')

parser.add_argument('--protocol', default='cross_view', type=str,
                    help='traiining protocol of ntu')

parser.add_argument('--finetune-skeleton-representation', default='graph-based', type=str,
                    help='which skeleton-representation to use for downstream training')
parser.add_argument('--pretrain-skeleton-representation', default='seq-based_and_graph-based', type=str,
                    help='which skeleton-representation where used for  pre-training')



# initilize weight
def weights_init_gru(model):
    with torch.no_grad():
        for child in list(model.children()):
            print("init ", child)
            for param in list(child.parameters()):
                if param.dim() == 2:
                    nn.init.xavier_uniform_(param)
    print('PC weight initial finished!')


def load_moco_encoder_q(model, pretrained):
    if os.path.isfile(pretrained):
        print("=> loading checkpoint '{}'".format(pretrained))
        checkpoint = torch.load(pretrained, map_location="cpu")

        # rename moco pre-trained keys
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                # remove prefix
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=False)
        # print("message", msg)
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

        print("=> loaded pre-trained model '{}'".format(pretrained))
    else:
        print("=> no checkpoint found at '{}'".format(pretrained))


def load_moco_encoder_r(model, pretrained):
    if os.path.isfile(pretrained):
        print("=> loading checkpoint '{}'".format(pretrained))
        checkpoint = torch.load(pretrained, map_location="cpu")

        # rename moco pre-trained keys
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # retain only encoder_r up to before the embedding layer
            if k.startswith('module.encoder_r') and not k.startswith('module.encoder_r.fc'):
                # remove prefix
                state_dict[k[len("module.encoder_r."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
            # if k.startswith('encoder_r') and not k.startswith('encoder_r.fc'):
            #     # remove prefix
            #     state_dict[k[len("encoder_r."):]] = state_dict[k]
            # # delete renamed or unused k
            # del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=False)
        # print("message", msg)
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

        print("=> loaded pre-trained model '{}'".format(pretrained))
    else:
        print("=> no checkpoint found at '{}'".format(pretrained))


def load_pretrained(args, model):
    if args.pretrain_skeleton_representation == 'seq-based' or args.pretrain_skeleton_representation == 'image-based' or args.pretrain_skeleton_representation == 'graph-based':

        load_moco_encoder_q(model, args.pretrained)

    # inter-skeleton contrastive  pretrianing
    else:
        if args.finetune_skeleton_representation == 'seq-based' and (
                args.pretrain_skeleton_representation == 'seq-based_and_graph-based' or args.pretrain_skeleton_representation == 'seq-based_and_image-based'):
            # load  only seq-based query encoder of the inter-skeleton framework  pretrained using seq-based_and_graph-based or 'seq-based_and_image-based' representations
            load_moco_encoder_q(model, args.pretrained)

        elif args.finetune_skeleton_representation == 'graph-based' and args.pretrain_skeleton_representation == 'seq-based_and_graph-based':
            # load  only graph-based query encoder of the inter-skeleton framework pretrained using seq-based_and_graph-based representations
            load_moco_encoder_r(model, args.pretrained)

        elif args.finetune_skeleton_representation == 'graph-based' and args.pretrain_skeleton_representation == 'graph-based_and_image-based':
            # load  only graph-based query encoder of the inter-skeleton framework pretrained using graph-based_and_image-based representations
            load_moco_encoder_q(model, args.pretrained)



def kmeanstraining(data_train,  nc=120):
    print("Number of classes = ", nc)
    Xtr_Norm = preprocessing.normalize(data_train)
    kmeans = KMeans(n_clusters=nc, random_state=0).fit(Xtr_Norm)

    return kmeans



def traindata_extract_hidden(model, data_train):
    for ith, (ith_data, label, number_of_frame) in enumerate(data_train):
        input_tensor = ith_data.to(device)

        en_hi = model(input_tensor, knn_eval=True)
        en_hi = en_hi.squeeze()

        if ith == 0:
            hidden_array_train = en_hi[:, :].detach().cpu().numpy()

        else:
            hidden_array_train = np.vstack((hidden_array_train, en_hi[:, :].detach().cpu().numpy()))


    return hidden_array_train


def  attackbykmeans( kmeans, model, data_eval):
    for ith, (ith_data, label, number_of_frame) in enumerate(data_eval):
        model.eval()
        prednew1 = []
        prednew2 = []
        input_tensor = ith_data.to(device)
        input = input_tensor.clone()
        momentum = torch.zeros_like(input).detach().cuda()

        # find cluster center
        ep = 400
        en_hi = model(input_tensor, knn_eval=True)
        en_hi = en_hi.squeeze()
        features = en_hi.detach().cpu().numpy()
        Xte_Norm = preprocessing.normalize(features)
        cluscen = kmeans.cluster_centers_
        precen1 = np.zeros((8,110,256))
        precen2 = Xte_Norm.copy()

        # sort samples
        for i in range(len(label)):
            precen2[[0, i]] = precen2[[i, 0]]
            x = precen2.copy()
            precen2 = Xte_Norm.copy()
            prednew2.append(x)
        pred2 = np.array(prednew2)
        # sort centers
        for i in range(len(label)):
            a = np.expand_dims(precen2[i, :], 0).repeat(120,axis=0)
            idx = (np.sqrt(np.sum(np.square(a - cluscen), axis=-1)))

            max_index1 = np.argsort(idx)
            max_index1 = max_index1[10:]
            prednew1.append(max_index1)
        pred1 = np.array(prednew1)

        for i in range(len(label)):
            precen1[i, :] = cluscen[pred1[i], :]

        precen1 = torch.tensor(precen1)
        precen2 = torch.tensor(pred2)
        precencat = torch.cat((precen2, precen1), dim=1).cuda()
        precencat =precencat.cuda()
        eps = 0.01
        c = torch.zeros((8, 3, 1, 25, 2)).cuda()
        # read the TV-AR parameters
        path = '../ARmodel/NTU{}CVval.npz'.format(ith)
        data = np.load(path)
        Ar1 = data['AR1']
        Ar2 = data['AR2']
        vel = torch.from_numpy(Ar2).cuda()
        vel1 = torch.from_numpy(Ar1).cuda()
        input.requires_grad = True
        # start iterative attack
        for i in range(ep):
            model.eval()
            en_hiatt = model(input, knn_eval=True)
            en_hiatt = en_hiatt.squeeze()
            en_hiatt = torch.nn.functional.normalize(en_hiatt, p=2, dim=1)
            en_hiatt = en_hiatt.unsqueeze(1).repeat(1,118,1)
            cos = nn.CosineSimilarity(dim=-1, eps=1e-1)
            simloss = cos(en_hiatt, precencat)
            labels = torch.tensor([0] * len(label)).long().cuda()
            loss = -1*nn.CrossEntropyLoss()(simloss, labels)

            input.grad = None
            input.retain_grad()
            loss.backward(retain_graph=True)
            cgs = input.grad
            cgs = cgs / torch.mean(torch.abs(cgs), dim=(1,2,3,4), keepdim=True)


            # calculate SMI-gradient(S2MI-FGSM)
            change = cgs[:, :, 1:, :, :]
            changes = torch.cat((change,c),dim=2)
            change1 = cgs[:, :, 2:, :, :]
            changes1 = torch.cat((change1,c,c),dim=2)
            cgs = cgs + momentum * 1 + vel1 * changes + vel * changes1
            # cgs = cgs  + vel1 * changes + vel * changes1
            momentum = cgs


            cgs = cgs.sign()
            input = input - 1./10000 * cgs
            # Budgets
            for k in range(len(label)):
                double = torch.nonzero(input_tensor[k, :, :, :, 1])
                if double.size() == torch.Size([0, 3]):
                    input[k, :, :, :, 1] = input_tensor[k, :, :,:,  1]
            input = input.to(torch.float32)
            input = torch.where(input > input_tensor + eps, input_tensor + eps, input)
            input = torch.where(input < input_tensor - eps, input_tensor - eps, input)
            if i % 25 == 0 or i == 0 or i == ep - 1:
                print(loss,  i)
            # save adversarial samples
            if i == ep-1:
                np.savez_compressed('./Samples/NTU%d.npz' % (ith),
                                    clips=ith_data.cpu().detach().numpy(), oriClips=input.cpu().detach().numpy(),
                                    labels=label.cpu().detach().numpy())








    return  input



def nobox_attack(model, eval_loader):
    model.eval()
    hi_eval= traindata_extract_hidden(model, eval_loader)
    kmeansmodel = kmeanstraining(hi_eval, nc=120)
    result = attackbykmeans(kmeansmodel, model, eval_loader)

    return result



def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

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

    # training dataset
    from options import options_attack as options
    if args.finetune_dataset == 'ntu60' and args.protocol == 'cross_view':
        opts = options.opts_ntu_60_cross_view()
    elif args.finetune_dataset == 'ntu60' and args.protocol == 'cross_subject':
        opts = options.opts_ntu_60_cross_subject()


    opts.test_feeder_args['input_representation'] = args.finetune_skeleton_representation

    # create model
    if args.finetune_skeleton_representation == 'seq-based':
        # Gru model
        model = BIGRU(**opts.bi_gru_model_args)

        print("options", opts.bi_gru_model_args, opts.train_feeder_args, opts.test_feeder_args)
        if not args.pretrained:
            weights_init_gru(model)

    elif args.finetune_skeleton_representation == 'graph-based':
        model = AGCN(**opts.agcn_model_args)
        print("options", opts.agcn_model_args, opts.train_feeder_args, opts.test_feeder_args)



    # load from pre-trained  model
    load_pretrained(args, model)

    if args.gpu is not None:
        model = model.cuda()


    cudnn.benchmark = True

    ## Data loading code

    val_dataset = get_finetune_validation_set(opts)


    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    results = nobox_attack(model, val_loader)



if __name__ == '__main__':
    main()
