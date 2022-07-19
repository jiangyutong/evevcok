
import os
import sys
from pprint import pprint
import numpy as np
import datetime
import torch
import torch.nn as nn
import torch.optim

from torch.autograd import Variable
import torch.nn.functional as F
import libs.parser.parse as parse
import libs.dataset.h36m.cameras as cameras
from opt import Options
import src.utils as utils
import src.log as log

from src.model2d import CVAE_Linear, weight_init
H36M_NAMES = ['']*32
H36M_NAMES[0]  = 'Hip'
H36M_NAMES[1]  = 'RHip'
H36M_NAMES[2]  = 'RKnee'
H36M_NAMES[3]  = 'RFoot'
H36M_NAMES[6]  = 'LHip'
H36M_NAMES[7]  = 'LKnee'
H36M_NAMES[8]  = 'LFoot'
H36M_NAMES[12] = 'Spine'
H36M_NAMES[13] = 'Thorax'
H36M_NAMES[14] = 'Neck/Nose'
H36M_NAMES[15] = 'Head'
H36M_NAMES[17] = 'LShoulder'
H36M_NAMES[18] = 'LElbow'
H36M_NAMES[19] = 'LWrist'
H36M_NAMES[25] = 'RShoulder'
H36M_NAMES[26] = 'RElbow'
H36M_NAMES[27] = 'RWrist'

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



def project_point_radial(P, R, T, f, c, k, p):
    """
    Project points from 3d to 2d using camera parameters
    including radial and tangential distortion

    Args
        P: Nx3 points in world coordinates
        R: 3x3 Camera rotation matrix
        T: 3x1 Camera translation parameters
        f: (scalar) Camera focal length
        c: 2x1 Camera center
        k: 3x1 Camera radial distortion coefficients
        p: 2x1 Camera tangential distortion coefficients
    Returns
        Proj: Nx2 points in pixel space
        D: 1xN depth of each point in camera space
        radial: 1xN radial distortion per point
        tan: 1xN tangential distortion per point
        r2: 1xN squared radius of the projected points before distortion
    """
    # P is a matrix of 3-dimensional points
    assert len(P.shape) == 2
    assert P.shape[1] == 3

    N = P.shape[0]
    X = R.dot(P.T - T)  # rotate and translate
    XX = X[:2, :] / X[2, :]
    r2 = XX[0, :] ** 2 + XX[1, :] ** 2

    radial = 1 + np.einsum('ij,ij->j', np.tile(k, (1, N)),
                           np.array([r2, r2 ** 2, r2 ** 3])
                           );
    tan = p[0] * XX[1, :] + p[1] * XX[0, :]

    XXX = (XX * np.tile(radial + tan, (2, 1))
           + np.outer(np.array([p[1], p[0]]).reshape(-1), r2))

    Proj = (f * XXX) + c
    Proj = Proj.T

    D = X[2,]
    return Proj, D, radial, tan, r2


def project_to_cameras(poses_set, cams, ncams=4):
    """
    Project 3d poses using camera parameters

    Args
        poses_set: dictionary containing 3d poses
        cams: dictionary containing camera parameters
        ncams: number of cameras per subject
    Returns
        t2d: dictionary with 2d poses
    """
    t2d = {}
    for t3dk in sorted(poses_set.keys()):
        subj, a, seqname = t3dk
        t3d = poses_set[t3dk]
        for cam in range(ncams):
            R, T, f, c, k, p, name = cams[(subj, cam+1)]
            pts2d, _, _, _, _ = cameras.project_point_radial(np.reshape(t3d, [-1, 3]), R, T, f, c, k, p)
            pts2d = np.reshape(pts2d, [-1, len(H36M_NAMES)*2])
            t2d[ t3dk ] = pts2d
    return t2d

if __name__ == '__main__':

    opt = parse.parse_arg()  # 不是cvae
    option = Options().parse()  # cvae
    opt.gpuid=2
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
    epcho = 200
    train_dict_3d_gen = {}
    optimizer = torch.optim.Adam(mymodel.parameters(), lr=0.01, weight_decay=option.lr_decay)
        # update summary
        # if (i % 100 == 0):
    inputs_get = {}
    # for key in train_dict_3d.keys():
    #     targets = train_dict_3d[key]
    #     mymean = np.mean(targets, axis=0)
    #     mymean = np.mean(mymean, axis=0)
    #     mystd = np.std(targets, axis=0)
    #     mystd = np.std(mystd, axis=0)
    #     # 7、创建一个3x3的，均值为0，方差为1,正太分布的随即数数组
    #     inputs = np.random.normal(mymean, mystd, targets.shape)
    #     inputs_get[key]=inputs
    cameras_path = os.path.join(opt.data_dir_cvae, 'cameras.npy')
    rcams = np.load(cameras_path, allow_pickle=True).item()
    train_dict_2d_targ = project_to_cameras(train_dict_3d, rcams, ncams=4)
    for i in range(1,epcho+1):
        j=0
        # sys.stdout.flush()

        for key in train_dict_3d.keys():
            targets = train_dict_3d[key]
            # 7、创建一个3x3的，均值为0，方差为1,正太分布的随即数数组
            inputs=train_dict_2d_targ[key]
            inputs=torch.Tensor(inputs)
            targets = torch.Tensor(targets)
            if option.cuda:

                inputs = Variable(inputs.cuda())
                targets = Variable(targets.cuda())


            mymodel.train()
            l2_loss, cvae_loss, gsnn_loss, kl_loss = utils.AverageMeter(), utils.AverageMeter(), utils.AverageMeter(), utils.AverageMeter()
            if i % option.lr_decay == 0 or i == 1:
                lr_now = utils.lr_decay(optimizer, i, option.lr,  option.lr_decay, option.lr_gamma)

            # forward pass
            out_cvae, out_gsnn, post_mu, post_logvar = mymodel(inputs, targets)

            # backward pass
            optimizer.zero_grad()
            loss= F.smooth_l1_loss(out_cvae,targets)

            loss.backward()
            optimizer.step()
            if  key==(1, 'Directions', 'Directions 1.h5'):
                print('epcho:{} key:{}   loss: {}'.format(i,key,loss))

            j=j+1
            if i == epcho:
                train_dict_3d_gen[key] = out_cvae
    now = datetime.datetime.now()
    stpath = str(now) + "_" + "jyt_geneticfirst_2d"
    save2Dpath = os.path.join(opt.data_dir_cvae, stpath)
    if not os.path.exists(save2Dpath):
        os.mkdir(save2Dpath)
    save2Dpath = save2Dpath+"/"+ "model"
    torch.save(mymodel, save2Dpath)
    save_path = "/media/zlz422/82BEDFE2BEDFCD33/jyt/project/EvSkeleton/data/human3.6M/jyt_geneticfirst_2d"
    np.save(save_path, cast_to_float(train_dict_3d_gen))
