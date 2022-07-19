
import os
import sys
from pprint import pprint
import numpy as np

import torch
import torch.nn as nn
import torch.optim

from torch.autograd import Variable
import torch.nn.functional as F
import libs.parser.parse as parse

from opt import Options
import src.utils as utils
import src.log as log
import datetime
from src.model2 import CVAE_Linear, weight_init

opt = parse.parse_arg()#不是cvae
option = Options().parse()#cvae
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
def loss_function(y, y_gsnn, x, mu, logvar):

    L2_cvae = F.mse_loss(y, x)


    return  L2_cvae


def cast_to_float(dic, dtype=np.float32):
    # cast to float 32 for space saving
    for key in dic.keys():
        dic[key] = dic[key].detach().cpu().numpy()
        dic[key] = dic[key].astype(dtype)
    return dic


if __name__ == '__main__':

    opt = parse.parse_arg()  # 不是cvae
    option = Options().parse()  # cvae
    option.cuda = opt.gpuid >= 0
    torch.cuda.set_device(opt.gpuid)
    mymodel = CVAE_Linear(option.cvaeSize, option.latent_size, option.numSamples_train, option.alpha,
                          option.cvae_num_stack)
    mymodel.apply(weight_init)
    if option.cuda:
        mymodel = mymodel.cuda()
    dict_path = os.path.join(opt.data_dir_cvae, 'threeDPose_train.npy')
    train_dict_3d = np.load(dict_path, allow_pickle=True).item()
    dict_path = os.path.join(opt.data_dir_cvae, 'threeDPose_test.npy')
    test_dict_3d = np.load(dict_path, allow_pickle=True).item()
    epcho = 10
    train_dict_3d_gen = {}
    optimizer = torch.optim.Adam(mymodel.parameters(), lr=0.001, weight_decay=option.lr_decay)
        # update summary
        # if (i % 100 == 0):
    inputs_get = {}
    for key in train_dict_3d.keys():
        targets = train_dict_3d[key]
        mymean = np.mean(targets, axis=0)
        mymean = np.mean(mymean, axis=0)
        mystd = np.std(targets, axis=0)
        mystd = np.std(mystd, axis=0)
        # 7、创建一个3x3的，均值为0，方差为1,正太分布的随即数数组
        inputs = np.random.normal(mymean, mystd, targets.shape)
        inputs_get[key]=inputs

    for i in range(1,epcho+1):
        mymodel.train()
        j=0
        # sys.stdout.flush()

        for key in train_dict_3d.keys():
            targets = train_dict_3d[key]

            # 7、创建一个3x3的，均值为0，方差为1,正太分布的随即数数组
            inputs=inputs_get[key]
            inputs=torch.Tensor(inputs)
            targets = torch.Tensor(targets)
            if option.cuda:

                inputs = Variable(inputs.cuda())
                targets = Variable(targets.cuda())



            l2_loss, cvae_loss, gsnn_loss, kl_loss = utils.AverageMeter(), utils.AverageMeter(), utils.AverageMeter(), utils.AverageMeter()
            # if i % option.lr_decay == 0 or i == 1:
            #     lr_now = utils.lr_decay(optimizer, i, option.lr,  option.lr_decay, option.lr_gamma)
                # backward pass
            optimizer.zero_grad()
            # forward pass
            out_cvae, out_gsnn, post_mu, post_logvar = mymodel(inputs, targets)


            loss =F.smooth_l1_loss(out_cvae,targets)
            # loss= loss_function(out_cvae, out_gsnn, targets, post_mu, post_logvar)

            loss.backward()
            optimizer.step()
            if key==(1, 'Directions', 'Directions 1.h5'):
                print('epcho:{} key:{}   loss: {}'.format(i,key,loss))

            j=j+1
            if i == epcho:
                train_dict_3d_gen[key] = out_cvae
    now = datetime.datetime.now()
    stpath = str(now) + "_" + "geneticfirst_random"
    save2Dpath = os.path.join(opt.data_dir_cvae, stpath)
    if not os.path.exists(save2Dpath):
        os.mkdir(save2Dpath)
    save2Dpath=save2Dpath+"/"+"model"
    torch.save(mymodel, save2Dpath)
    save_path = "/media/zlz422/82BEDFE2BEDFCD33/jyt/project/EvSkeleton/data/human3.6M/jyt_geneticfirst_random"
    np.save(save_path, cast_to_float(train_dict_3d_gen))
