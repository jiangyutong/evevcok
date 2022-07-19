"""
Utility functions for genetic evolution.
"""
import libs.dataset.h36m.cameras as cameras
import libs.dataset.h36m.data_utils as data_utils

from libs.skeleton.anglelimits import \
to_local, to_global, get_skeleton, to_spherical, \
nt_parent_indices, nt_child_indices, \
is_valid_local, is_valid

def show3Dposem(channels, ax, lcolor="#3498db", rcolor="#e74c3c", add_labels=True,
               gt=False, pred=False):  # blue, orange
    """
    Visualize a 3d skeleton

    Args
      channels: 96x1 vector. The pose to plot.
      ax: matplotlib 3d axis to draw on
      lcolor: color for left part of the body
      rcolor: color for right part of the body
      add_labels: whether to add coordinate labels
    Returns
      Nothing. Draws on ax.
    """

    assert channels.size == len(
        data_utils.H36M_NAMES) * 3, "channels should have 96 entries, it has %d instead" % channels.size
    vals = np.reshape(channels, (len(data_utils.H36M_NAMES), -1))

    I = np.array([1, 2, 3, 1, 7, 8, 1, 13, 14, 15, 14, 18, 19, 14, 26, 27]) - 1  # start points
    J = np.array([2, 3, 4, 7, 8, 9, 13, 14, 15, 16, 18, 19, 20, 26, 27, 28]) - 1  # end points
    LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
    # Make connection matrix
    for i in np.arange(len(I)):
        x, y, z = [np.array([vals[I[i], j], vals[J[i], j]]) for j in range(3)]
        if gt:
            if I[i] % 2 == 0:color = 'g'
            if I[i] % 3 == 0: color = 'b'
            if I[i]%4==0:color = 'm'
            if I[i]%5==0:color='c'
            ax.plot(x, z, -y, lw=4, c=color)
        #        ax.plot(x,y, z,  lw=2, c='k')

        elif pred:
            if i % 2 == 0: color = 'g'
            if i % 3 == 0: color = 'b'
            if i % 4 == 0: color = 'm'
            if i % 5 == 0: color = 'c'
            ax.plot(x, z, -y, lw=4, c=color)
        #        ax.plot(x,y, z,  lw=2, c='r')

        else:
            #        ax.plot(x,z, -y,  lw=2, c=lcolor if LR[i] else rcolor)
            if I[i] % 2 == 0:
                lcolor = 'salmon'
                rcolor = "darkorange"
            if I[i] % 3 == 0:
                lcolor = 'darkorange'
                rcolor = "darkorange"
            if I[i] % 4 == 0:
                lcolor = 'm'
                rcolor = "m"
            if I[i] % 5 == 0:
                lcolor = 'red'
                rcolor = "red"
            if I[i] == 0:
                lcolor = "lightcoral"
                rcolor = "lightcoral"
            if I[i]==7:
                lcolor = "r"
                rcolor = "r"
            if I[i]  == 2:
                lcolor = 'gold'
                rcolor = "gold"
            if I[i] == 14:

                rcolor = "lightcoral"
            if I[i] == 8:
                rcolor = "m"
            # if I[i]==12:
            #     lcolor = "lightcoral"
            #     rcolor = "lightcoral"
            # if I[i] == 19:
            #     lcolor = "lightcoral"
            #     rcolor = "lightcoral"
            ax.plot(x, y, z, lw=3, c=lcolor if LR[i] else rcolor)

    RADIUS = 750  # space around the subject
    xroot, yroot, zroot = vals[0, 0], vals[0, 1], vals[0, 2]
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])


    if add_labels:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    # Get rid of the ticks and tick labels
    #  ax.set_xticks([])
    #  ax.set_yticks([])
    #  ax.set_zticks([])
    #
    #  ax.get_xaxis().set_ticklabels([])
    #  ax.get_yaxis().set_ticklabels([])
    #  ax.set_zticklabels([])
    ax.set_aspect('auto')

    # Get rid of the panes (actually, make them white)
    white = (1.0, 1.0, 1.0, 0.0)
    ax.w_xaxis.set_pane_color(white)
    ax.w_yaxis.set_pane_color(white)
    # Keep z pane

    # Get rid of the lines in 3d
    ax.w_xaxis.line.set_color(white)
    ax.w_yaxis.line.set_color(white)
    ax.w_zaxis.line.set_color(white)
    ax.set_axis_off()


def re_orderm(skeleton):
    skeleton = skeleton.copy().reshape(-1, 3)
    # permute the order of x,y,z axis
    skeleton[:, [0, 1, 2]] = skeleton[:, [0, 2, 1]]
    return skeleton.reshape(96)


def plot_3d_ax(ax,
               elev,
               azim,
               pred,
               title=None
               ):
    #   elev=10.,
    #                azim=-90,
    ax.view_init(elev=-180, azim=84)
    show3Dposem(re_orderm(pred), ax)
    plt.title(title)
    return
def adjust_figure(left = 0,
                  right = 1,
                  bottom = 0.01,
                  top = 0.95,
                  wspace = 0,
                  hspace = 0.4
                  ):
    plt.subplots_adjust(left, bottom, right, top, wspace, hspace)
    return
import matplotlib.pyplot as plt
import os
import logging
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

root = "../resources/constraints"
# Joints in H3.6M -- data has 32 joints, but only 17 that move
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
total_joints_num = len(H36M_NAMES)

# this dictionary stores the parent indice for each joint
# key:value -> child joint index:its parent joint index
parent_idx = {1:0, 2:1, 3:2, 6:0, 7:6, 8:7, 12:0, 13:12, 14:13, 15:14, 17:13, 
              18:17, 19:18, 25:13, 26:25, 27:26
              }

# this dictionary stores the children indices for each parent joint
# key:value -> parent index: joint indices for its children as a list  
children_idx = {
        0: [1, 6],
        1: [2], 2: [3],
        6: [7], 7: [8],
        13: [14, 17, 25],
        14: [15], 17: [18], 18:[19],
        25: [26], 26:[27]
        }

# used roots for random selection
root_joints = [0, 1, 2, 6, 7, 13, 17, 18, 25, 26]
# names of the bone vectors attached on the human torso
bone_name = {
 1: 'thorax to head top',
 2: 'left shoulder to left elbow',
 3: 'left elbow to left wrist',
 4: 'right shoulder to right elbow',
 5: 'right elbow to right wrist',
 6: 'left hip to left knee',
 7: 'left knee to left ankle',
 8: 'right hip to right knee',
 9: 'right knee to right ankle'
}    
# this dictionary stores the sub-tree rooted at each root joint
# key:value->root joint index:list of bone vector indices 
bone_indices = {0: [5, 6, 7, 8],
                1: [7, 8],
                2: [8],
                6: [5, 6],
                7: [6],
                13: [1, 2, 3, 4], # thorax
                17: [1, 2],
                18: [2],
                25: [3, 4],
                26: [4]
                }
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# _posefrom __future__ import print_function, absolute_import, division

import os
import sys
from pprint import pprint
import numpy as np

import torch
import torch.nn as nn
import torch.optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F

from opt import Options
import src.utils as utils
import src.log as log

from src.model import CVAE_Linear, weight_init
from src.datasets.human36m2 import Human36M

option = Options().parse()
def loss_function(y, y_gsnn, x, mu, logvar):

    L2_cvae = option.alpha * F.mse_loss(y, x)
    L2_gsnn = (1 - option.alpha) * F.mse_loss(y_gsnn, x)
    L2 = L2_cvae + L2_gsnn
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return L2, L2_cvae, L2_gsnn, KLD


keysall=[]
def test_multiposenet(test_loader, model, criterion, stat_3d, stat_2d, procrustes=False):
    model.eval()
    l2_loss = utils.AverageMeter()

    # global error trackers
    all_dist, all_dist_samples, all_dist_ordsamp_weighted, all_dist_ordsamp_weighted_pred = [], [], {}, {}
    temp_gt = np.linspace(0.1, 1, num=10) # range of temperatures for softmax in OrdinalScore
    temp_pred = np.linspace(0.1, 1, num=10)
    for ind, t in enumerate(temp_gt):
        all_dist_ordsamp_weighted[ind] = []
        all_dist_ordsamp_weighted_pred[ind] = []

    for i, (inps, tars, ordinals,keys) in enumerate(test_loader):

        if (not i % 20 == 0): # for quick validation during training
            continue
        inputs = Variable(inps.cuda())
        targets = Variable(tars.cuda(async=True))

        batch_size = inputs.shape[0]
        z_samples, out_samples = [], []
        num_samples = option.numSamples
        # generate sample set
        for j in range(num_samples):
            z = torch.randn(batch_size, option.latent_size).cuda()
            z_samples.append(z)

            out = model.decode(z, inputs)
            out_samples.append(out)

            loss_l2, _, _, loss_kl = loss_function(out, out, targets,
                                                   torch.zeros((option.test_batch, option.latent_size)).cuda(),
                                                   torch.zeros((option.test_batch, option.latent_size)).cuda())
            loss_l2 = loss_l2 * option.weight_l2
            loss = loss_kl + loss_l2

            l2_loss.update(loss_l2.item(), inputs.size(0))
            keysall.append(keys)
        out_samp = torch.cat([torch.unsqueeze(out_sample, dim=0) for out_sample in out_samples])
        out_mean = torch.mean(out_samp, dim=0)
        tars = targets

        outputs_samples_unnorm = np.vstack([utils.unNormalizeData(out_sample.data.cpu().numpy(), stat_3d['mean'], stat_3d['std'], stat_3d['dim_use'])[None] for out_sample in out_samples])


    outputs_samples_unnorm =outputs_samples_unnorm.reshape(-1,96)
    return outputs_samples_unnorm,keysall
def modify_pose(skeleton, local_bones, bone_length, ro=False):
    # get a new pose by modify an existing pose with input local bone vectors
    # and bone lengths
    new_bones = to_global(skeleton, local_bones)['bg']
    new_pose = get_skeleton(new_bones, skeleton, bone_length=bone_length)
    if ro:
        new_pose = re_order(new_pose)
    return new_pose.reshape(-1)
def cvae():
    opt = Options().parse()
    start_epoch = 0
    err_best = 1000
    glob_step = 0
    lr_now = opt.lr

    # save options
    log.save_options(opt, opt.ckpt)

    # create model
    print(">>> creating model")
    model = CVAE_Linear(opt.cvaeSize, opt.latent_size, opt.numSamples_train, opt.alpha, opt.cvae_num_stack)
    model.cuda()
    model.apply(weight_init)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    criterion = nn.MSELoss(size_average=True).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    # load ckpt
    if opt.load:
        print(">>> loading ckpt from '{}'".format(opt.load))
        ckpt = torch.load(opt.load)
        start_epoch = ckpt['epoch']
        err_best = ckpt['err']
        glob_step = ckpt['step']
        lr_now = ckpt['lr']
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        print(">>> ckpt loaded (epoch: {} | err: {})".format(start_epoch, err_best))
    if opt.resume:
        logger = log.Logger(os.path.join(opt.ckpt, 'log.txt'), resume=True)
    else:
        logger = log.Logger(os.path.join(opt.ckpt, 'log.txt'))
        logger.set_names(['epoch', 'lr', 'loss_train', 'loss_test', 'err_mean', 'err_bestsamp'])

    # list of action(s)
    actions = utils.define_actions('All')
    num_actions = len(actions)
    print(">>> actions to use (total: {}):".format(num_actions))
    pprint(actions, indent=4)
    print(">>>")
    # data loading
    print(">>> loading data")
    # stat_2d = torch.load(os.path.join(opt.data_dir, 'stat_2d.pth.pt'))
    # stat_3d = torch.load(os.path.join(opt.data_dir, 'stat_3d.pth.pt'))
    stat_2d = torch.load("../data/stat_2d.pth.pt")
    stat_3d = torch.load("../data/stat_3d.pth.pt")
    train_loader = DataLoader(
        dataset=Human36M(actions=actions, data_path=opt.data_dir, procrustes=opt.procrustes),
        batch_size=opt.train_batch,
        shuffle=True,
        num_workers=opt.job, )

    print(">>> data loaded !")

    cudnn.benchmark = True
    outputs_samples_unnorm,keysall = test_multiposenet(train_loader, model, criterion, stat_3d, stat_2d,procrustes=opt.procrustes)

    return  outputs_samples_unnorm,keysall
def unnormalize(data, mean, std):
    return (data*std) + mean

def postprocess_3d( poses):
    return poses - np.tile( poses[:,:3], [1, len(H36M_NAMES)] )

def calc_errors(pred_poses, gt_poses, protocol='mpjpe'):
    # error after a regid alignment, corresponding to protocol #2 in the paper
    # Compute Euclidean distance error per joint
    sqerr = (pred_poses - gt_poses)**2 # Squared error between prediction and expected output
    sqerr = sqerr.reshape(len(sqerr), -1, 3)
    sqerr = np.sqrt(sqerr.sum(axis=2))
    if protocol == 'mpjpe':
        ret = sqerr.mean(axis=1)
        ret = ret.reshape(len(ret), 1)
    else:
        raise NotImplementedError
    return ret

def to_numpy(tensor):
    return tensor.data.cpu().numpy()

def cast_to_float(dic, dtype=np.float32):
    # cast to float 32 for space saving
    for key in dic.keys():
        dic[key] = dic[key].astype(dtype)
    return dic

# global variables
parent_idx = [0, 6, 7, \
              0, 1, 2, \
              0, 12, 13, 14,\
              13, 17, 18,\
              13, 25, 26]
child_idx = [6, 7, 8, \
             1, 2, 3, \
             12, 13, 14, 15,\
             17, 18, 19,\
             25, 26, 27]
def get_random_rotation(sigma=60.):
    angle = np.random.normal(scale=sigma)
    axis_idx = np.random.choice(3, 1)
    if axis_idx == 0:
        r = R.from_euler('xyz', [angle, 0., 0.], degrees=True)
    elif axis_idx == 1:
        r = R.from_euler('xyz', [0., angle, 0.], degrees=True)
    else:
        r = R.from_euler('xyz', [0., 0., angle], degrees=True)
    return r

def rotate_bone_random(bone, sigma=10.):
    r = get_random_rotation(sigma)
    bone_rot = r.as_dcm() @ bone.reshape(3,1)
    return bone_rot.reshape(3)
def get_bone_length(skeleton):
    """
    Compute limb length for a given skeleton.
    """
    bones = skeleton[nt_parent_indices, :] - skeleton[nt_child_indices, :]
    bone_lengths = to_spherical(bones)[:, 0]
    return bone_lengths
def re_order(skeleton):
    # the ordering of coordinate used by the Prior was x,z and y
    return skeleton[:, [0,2,1]]
def rotate_pose_random(pose=None, sigma=60.):
    # pose shape: [n_joints, 3]
    if pose is None:
        result = None
    else:
        r = get_random_rotation()
        pose = pose.reshape(32, 3)
        # rotate around hip
        hip = pose[0].reshape(1, 3)
        relative_pose = pose - hip
        rotated = r.as_dcm() @ relative_pose.T
        result = rotated.T + hip
    return result
def set_z(pose, target):
    if pose is None:
        return None
    original_shape = pose.shape
    pose = pose.reshape(32, 3)
    min_val = pose[:, 2].min()
    pose[:, 2] -= min_val - target
    return pose.reshape(original_shape)


def show3Dpose(channels,
               ax,
               lcolor="#3498db",
               rcolor="#e74c3c",
               add_labels=True,
               gt=False,
               pred=False,
               plot_dot=False
               ):  # blue, orange
    """
    Visualize a 3d skeleton

    Args
      channels: 96x1 vector. The pose to plot.
      ax: matplotlib 3d axis to draw on
      lcolor: color for left part of the body
      rcolor: color for right part of the body
      add_labels: whether to add coordinate labels
    Returns
      Nothing. Draws on ax.
    """

    if channels.shape[0] == 96:
        vals = np.reshape(channels, (32, -1))
    else:
        vals = channels
    I = np.array([1, 2, 3, 1, 7, 8, 1, 13, 14, 15, 14, 18, 19, 14, 26, 27]) - 1  # start points
    J = np.array([2, 3, 4, 7, 8, 9, 13, 14, 15, 16, 18, 19, 20, 26, 27, 28]) - 1  # end points
    LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
    dim_use_3d = [3, 4, 5, 6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23, 24, 25,
                  26, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 51, 52, 53,
                  54, 55, 56, 57, 58, 59, 75, 76, 77, 78, 79, 80, 81, 82, 83]
    # Make connection matrix
    for i in np.arange(len(I)):
        x, y, z = [np.array([vals[I[i], j], vals[J[i], j]]) for j in range(3)]
        if gt:
            ax.plot(x, y, z, lw=4, c='k')
        #        ax.plot(x,y, z,  lw=2, c='k')

        elif pred:
            ax.plot(x, z, -y, lw=4, c='r')
        #        ax.plot(x,y, z,  lw=2, c='r')

        else:
            #        ax.plot(x,z, -y,  lw=2, c=lcolor if LR[i] else rcolor)
            ax.plot(x, y, z, lw=4, c=lcolor if LR[i] else rcolor)
    if plot_dot:
        joints = channels.reshape(96)
        joints = joints[dim_use_3d].reshape(16, 3)
        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c='k', marker='o')
    RADIUS = 750  # space around the subject
    xroot, yroot, zroot = vals[0, 0], vals[0, 1], vals[0, 2]
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])

    if add_labels:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    ax.set_aspect('auto')
    #  ax.set_xticks([])
    #  ax.set_yticks([])
    #  ax.set_zticks([])

    #  ax.get_xaxis().set_ticklabels([])
    #  ax.get_yaxis().set_ticklabels([])
    #  ax.set_zticklabels([])
    # Get rid of the panes (actually, make them white)
    #  white = (1.0, 1.0, 1.0, 0.0)
    #  ax.w_xaxis.set_pane_color(white)
    #  ax.w_yaxis.set_pane_color(white)
    # Keep z pane

    # Get rid of the lines in 3d
    #  ax.w_xaxis.line.set_color(white)
    #  ax.w_yaxis.line.set_color(white)
    #  ax.w_zaxis.line.set_color(white)
    ax.view_init(0, 90)
    plt.show()
def show3Dpose2(channels,
               ax,
               lcolor="#3498db",
               rcolor="#e74c3c",
               add_labels=True,
               gt=False,
               pred=False,
               plot_dot=False
               ):  # blue, orange
    """
    Visualize a 3d skeleton

    Args
      channels: 96x1 vector. The pose to plot.
      ax: matplotlib 3d axis to draw on
      lcolor: color for left part of the body
      rcolor: color for right part of the body
      add_labels: whether to add coordinate labels
    Returns
      Nothing. Draws on ax.
    """

    if channels.shape[0] == 96:
        vals = np.reshape(channels, (32, -1))
    else:
        vals = channels
    I = np.array([1, 2, 3, 1, 7, 8, 1, 13, 14, 15, 14, 18, 19, 14, 26, 27]) - 1  # start points
    J = np.array([2, 3, 4, 7, 8, 9, 13, 14, 15, 16, 18, 19, 20, 26, 27, 28]) - 1  # end points
    LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
    dim_use_3d = [3, 4, 5, 6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23, 24, 25,
                  26, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 51, 52, 53,
                  54, 55, 56, 57, 58, 59, 75, 76, 77, 78, 79, 80, 81, 82, 83]
    # Make connection matrix
    for i in np.arange(len(I)):
        x, y, z = [np.array([vals[I[i], j], vals[J[i], j]]) for j in range(3)]
        if gt:
            ax.plot(x, y, z, lw=4, c='k')
        #        ax.plot(x,y, z,  lw=2, c='k')

        elif pred:
            ax.plot(x, z, -y, lw=4, c='r')
        #        ax.plot(x,y, z,  lw=2, c='r')

        else:
            #        ax.plot(x,z, -y,  lw=2, c=lcolor if LR[i] else rcolor)
            ax.plot(x, y, z, lw=4, c=lcolor if LR[i] else rcolor)
            ax.text(x[0],y[0],z[0],I[i])
            ax.text(x[1],y[1],z[1],J[i])
    if plot_dot:
        joints = channels.reshape(96)
        joints = joints[dim_use_3d].reshape(16, 3)
        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c='k', marker='o')
    RADIUS = 400  # space around the subject
    xroot, yroot, zroot = vals[0, 0], vals[0, 1], vals[0, 2]
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])

#     if add_labels:
#         ax.set_xlabel("x")
#         ax.set_ylabel("y")
#         ax.set_zlabel("z")

    ax.set_aspect('auto')
    ax.view_init(90, 90)
    plt.show()
def one_iteration(population, opt, model_file=None):
    """
    Run one iteration to produce the next generation.
    """
    post_processing = True

    offsprings,keysall = cvae()
    offsprings_r=offsprings
    # np.save("/media/zlz422/82BEDFE2BEDFCD33/jyt/project/EvSkeleton/data/jyt.npy",offsprings)
    tmpLeng = offsprings_r.shape[0]
    tmpi = 0
    for i in range(tmpLeng):
        offsprings_r_tmpi=offsprings_r[tmpi]
        offsprings_r_tmpi=offsprings_r_tmpi.reshape(total_joints_num, -1)
        offsprings_r_tmpi = re_order(offsprings_r_tmpi)
        offsprings_r_tmpi_length = get_bone_length(offsprings_r_tmpi)
        bones_offsprings_r_tmpi = to_local(offsprings_r_tmpi)
        # plt.figure()
        # plt.title('or')
        # ax1 = plt.subplot(1, 1, 1, projection='3d')
        # ax1.grid(False)
        # plt.axis('off')
        # show3Dpose2(offsprings_r_tmpi, ax1, add_labels=False, plot_dot=True)
        # plt.tight_layout()
        # if opt.M:
        #     # global mutation: rotate the whole 3D skeleton
        #     if np.random.rand() <= opt.MRG:
        #         offsprings_r_tmpi = rotate_pose_random(offsprings_r_tmpi, sigma=opt.SDG)
            # plt.figure()
            # plt.title('m')
            # ax1 = plt.subplot(1, 1, 1, projection='3d')
            # ax1.grid(False)
            # plt.axis('off')
            # show3Dpose2(offsprings_r_tmpi, ax1, add_labels=False, plot_dot=True)
            # plt.tight_layout()
        if opt.C:
            # apply joint angle constraint as the fitness function
            valid_vec_fa = is_valid_local(bones_offsprings_r_tmpi)
        if not opt.C or valid_vec_fa.sum() >= opt.Th:
            offsprings_r_tmpi = modify_pose(offsprings_r_tmpi, bones_offsprings_r_tmpi, offsprings_r_tmpi_length, ro=True)
            if post_processing:
                # move the poses to the ground plane
                set_z(offsprings_r_tmpi, np.random.normal(loc=20.0, scale=3.0))
            offsprings_r[tmpi] = offsprings_r_tmpi
        else:
            offsprings_r = np.delete(offsprings_r, tmpi, 0)
            tmpi = tmpi - 1


        # if opt.DE and opt.V:
        # if opt.V:
        #     plt.figure()
        #     ax1 = plt.subplot(1, 4, 1, projection='3d')
        #     plt.title('father')
        #     show3Dpose(offsprings_r[tmpi], ax1, add_labels=False, plot_dot=True)
        #     plt.tight_layout()
        tmpi = tmpi + 1
    #
    # if opt.Mer:
    #     # merge the offsprings with the parents
    #     # population = np.vstack([population, offsprings])
    population = np.vstack([population, offsprings_r])
    # else:
    #     population = offsprings
    return population

def get_save_path(opt, gen_idx):
    if opt.WS:
        save_path = os.path.join(opt.SD, opt.SS, opt.SN)
    else:
        save_path = os.path.join(opt.SD, 'S15678', opt.SN)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = os.path.join(save_path, 'generation_{:d}.npy'.format(gen_idx))
    return save_path

def split_and_save(final_poses, parameters, gen_idx):
    temp_subject_list = [1, 5, 6, 7, 8]
    train_set_3d = {}
    poses_list = np.array_split(final_poses, len(temp_subject_list))
    for subject_idx in range(len(temp_subject_list)):
        train_set_3d[(temp_subject_list[subject_idx], 'n/a', 'n/a')] =\
        poses_list[subject_idx]         
    save_path = get_save_path(parameters, gen_idx)
    np.save(save_path, cast_to_float(train_set_3d))     
    print('file saved at {:s}!'.format(save_path))
    return

def save_results(poses, opt, gen_idx):

    split_and_save(poses, opt, gen_idx)
    split_and_save(poses, opt, gen_idx)
    return

def evolution(initial_population, opt, model_file=None):
    """
    Dataset evolution.
    """
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s]: %(message)s"
                        )
    population = initial_population
    save_results(initial_population, opt, 0)
    initial_num = len(initial_population)
    for gen_idx in range(1, opt.G+1):
        population = one_iteration(population, opt, model_file=model_file)
    save_results(population, opt, gen_idx)
    # if not enough
    enough =0
    if opt.E and len(population) < initial_num * opt.T:
        logging.info('Running extra generations to synthesize enough data...')
        while len(population) < initial_num * opt.T:
            enough +=1
            if enough>300:break
            gen_idx += 1
            logging.info('Generation {:d}...'.format(gen_idx))
            population = one_iteration(population, opt, model_file=model_file)
            if opt.I:
                save_results(population.copy(), opt, gen_idx)
                logging.info('Generation {:d} saved.'.format(gen_idx))
    save_results(population, opt, gen_idx)
    logging.info('Final population saved.')
    return population
