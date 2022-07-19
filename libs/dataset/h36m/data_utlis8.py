"""
Utility functions for dealing with Human3.6M dataset.
Some functions are adapted from https://github.com/una-dinosauria/3d-pose-baseline
"""
import datetime
import os
import numpy as np
import copy
import logging
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.mplot3d import Axes3D

import libs.dataset.h36m.cameras as cameras
import libs.dataset.h36m.pth_dataset as dataset
import libs.model.model2d6  as model
import libs.parser.parse as parse

import torch.nn.functional as F
# Human3.6m IDs for training and testing
TRAIN_SUBJECTS = [1, 5, 6, 7, 8]
TEST_SUBJECTS  = [9, 11]

# Use camera coordinate system
camera_frame = True

# Joint names in H3.6M -- data has 32 joints, but only 17 that move;
# these are the indices.
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

parent_indices = np.array([0, 1, 2, 0, 6, 7, 0, 12, 13, 14, 13, 17, 18, 13, 25, 26])
children_indices = np.array([1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27])

# Stacked Hourglass produces 16 joints. These are the names.
SH_NAMES = ['']*16
SH_NAMES[0]  = 'RFoot'
SH_NAMES[1]  = 'RKnee'
SH_NAMES[2]  = 'RHip'
SH_NAMES[3]  = 'LHip'
SH_NAMES[4]  = 'LKnee'
SH_NAMES[5]  = 'LFoot'
SH_NAMES[6]  = 'Hip'
SH_NAMES[7]  = 'Spine'
SH_NAMES[8]  = 'Thorax'
SH_NAMES[9]  = 'Head'
SH_NAMES[10] = 'RWrist'
SH_NAMES[11] = 'RElbow'
SH_NAMES[12] = 'RShoulder'
SH_NAMES[13] = 'LShoulder'
SH_NAMES[14] = 'LElbow'
SH_NAMES[15] = 'LWrist'

# The .h5 suffix in pose sequence name is just inherited from the original
# naming convention. The '.sh' suffix means stacked hourglass key-point detector
# used in previous works. Here we just use '.sh' to represent key-points obtained
# from any heat-map regression model. We used high-resolution net instead of
# stacked-hourglass model.

def load_ckpt(opt):
    cascade = torch.load(os.path.join(opt.ckpt_dir, 'model.th'))
    stats = np.load(os.path.join(opt.ckpt_dir, 'stats.npy'), allow_pickle=True).item()
    if opt.cuda:
        cascade.cuda()
    return cascade, stats

def list_remove(list_a, list_b):
    """
    Fine all elements of a list A that does not exist in list B.

    Args
      list_a: list A
      list_b: list B
    Returns
      list_c: result
    """
    list_c = []
    for item in list_a:
        if item not in list_b:
            list_c.append(item)
    return list_c

def add_virtual_cams(cams, visualize=False):
    """
    Deprecated. Add virtual cameras.
    """
    # add more cameras to the scene
    #R, T, f, c, k, p, name = cams[ (1,1) ]
    # plot the position of human subjects
    old_cam_num = 4
    def add_coordinate_system(ax, origin, system, length=300, new=False):
        # draw a coordinate system at a specified origin
        origin = origin.reshape(3, 1)
        start_points = np.repeat(origin, 3, axis=1)
        # system: [v1, v2, v3]
        end_points = start_points + system*length
        color = ['g', 'y', 'k'] # color for v1, v2 and v3
        if new:
            color = ['b', 'r', 'g']
        def get_args(start_points, end_points):
            x = [start_points[0], end_points[0]]
            y = [start_points[1], end_points[1]]
            z = [start_points[2], end_points[2]]
            return x, y, z
        for i in range(3):
            x, y, z = get_args(start_points[:,i], end_points[:,i])
            ax.plot(x, y, z,  lw=2, c=color[i])
        return

    def get_new_camera(system, center, rotation = [0,0,90.]):
        from scipy.spatial.transform import Rotation as Rotation
        center = center.reshape(3, 1)
        start_points = np.repeat(center, 3, axis=1)
        end_points = start_points + system
        r = Rotation.from_euler('xyz', rotation, degrees=True)
        start_points_new = r.as_dcm() @ start_points
        end_points_new = r.as_dcm() @ end_points
        new_system = [(end_points_new[:,i] - start_points_new[:,i]).reshape(3,1) for i in range(3)]
        new_system = np.hstack(new_system)
        return new_system, start_points_new[:,0]


    # the new cameras are added by rotating one existing camera
    # TODO: more rotations
    new_cams = cams.copy()
    for key in cams.keys():
        subject, camera_idx = key
        if camera_idx != 1: # only rotate the first camera
            continue
        R, T, f, c, k, p, name = cams[key]
        angles = [80., 130., 270., 320.]
        for angle_idx in range(len(angles)):
            angle = angles[angle_idx]
            new_R, new_T = get_new_camera(R.T, T, [0., 0., angle])
            new_cams[(subject, old_cam_num + angle_idx + 1)]\
            = (new_R.T, new_T.reshape(3,1), f, c, k, p, name+'new'+str(angle_idx+1))
    # visualize cameras used
    if visualize:
        train_set_3d = np.load('../data/human3.6M/h36m/numpy/threeDPose_train.npy').item()
        test_set_3d = np.load('../data/human3.6M/h36m/numpy/threeDPose_test.npy').item()
        hips_train = np.vstack(list(train_set_3d.values()))
        hips_test = np.vstack(list(test_set_3d.values()))
        ax = plt.subplot(111, projection='3d')
        chosen = np.random.choice(len(hips_train), 1000, replace=False)
        chosen_hips = hips_train[chosen, :3]
        ax.plot(chosen_hips[:,0], chosen_hips[:,1], chosen_hips[:,2], 'bo')
        chosen = np.random.choice(len(hips_test), 1000, replace=False)
        chosen_hips = hips_test[chosen, :3]
        ax.plot(chosen_hips[:,0], chosen_hips[:,1], chosen_hips[:,2], 'ro')
        ax.set_xlabel("x");ax.set_ylabel("y");ax.set_zlabel("z")
        plt.title('Blue dots: Hip positions in the h36m training set. \
                  Red dots: testing set. \
                  Old camera coordinates: x-green, y-yellow, z-black \
                  New camera coordinates: x-blue, y-red, z-green')
        plt.pause(0.1)
        for key in new_cams.keys():
            R, T, f, c, k, p, name = new_cams[key]
            # R gives camera basis vectors row-by-row, T gives camera center
            if 'new' in name:
                new = True
            else:
                new = False
            add_coordinate_system(ax, T, R.T, new=new)
        RADIUS = 3000 # space around the subject
        xroot, yroot, zroot = 0., 0., 500.
        ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
        ax.set_zlim3d([-RADIUS+zroot, RADIUS+zroot])
        ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])
        ax.set_aspect("equal")
    return new_cams

def down_sample_training_data(train_dict, opt):
    """
    Down-sample the training data.

    Args
      train_dict: python dictionary contraining the training data
      opt: experiment options
    Returns
      train_dict/sampled_dict: a dictionary containing a subset of training data
    """
    if opt.ws_name in ['S1', 'S15', 'S156']:
        sub_list = [int(opt.ws_name[i]) for i in range(1, len(opt.ws_name))]
        keys_to_delete = []
        for key in train_dict.keys():
            if key[0] not in sub_list:
                keys_to_delete.append(key)
        for key in keys_to_delete:
            del train_dict[key]
        return train_dict
    elif opt.ws_name in ['0.001S1','0.01S1', '0.05S1', '0.1S1', '0.5S1']:
        ratio = float(opt.ws_name.split('S')[0])
        # randomly sample a portion of 3D data
        sampled_dict = {}
        for key in train_dict.keys():
            if key[0] != 1:
                continue
            total = len(train_dict[key])
            sampled_num = int(ratio*total)
            chosen_indices = np.random.choice(total, sampled_num, replace=False)
            sampled_dict[key] = train_dict[key][chosen_indices].copy()
        return sampled_dict
    else:
        raise ValueError('Unknown experiment setting.')

def get_train_dict_3d(opt):
    """
    Get the training 3d skeletons as a Python dictionary.

    Args
      opt: experiment options
    Returns
      train_dict_3d: a dictionary containing training 3d poses
    """
    if not opt.train:
        return None
    dict_path = os.path.join(opt.data_dir, 'threeDPose_train.npy')
    #=========================================================================#
    # For real 2D detections, the down-sampling and data augmentation
    # are done later in get_train_dict_2d
    if opt.twoD_source != 'synthetic':
        train_dict_3d = np.load(dict_path, allow_pickle=True).item()
        return train_dict_3d
    #=========================================================================#
    # For synthetic 2D detections (For Protocol P1*), the down-sampling is
    # performed here and the data augmentation is assumed to be already done
    if opt.evolved_path is not None:
        # the data is pre-augmented
        train_dict_3d = np.load(opt.evolved_path, allow_pickle=True).item()
    elif opt.ws:
        # raw training data from Human 3.6M (S15678)
        # Down-sample the raw data to simulate an environment with scarce
        # training data, which is used in weakly-supervised experiments
        train_dict_3d = np.load(dict_path, allow_pickle=True).item()
        train_dict_3d = down_sample_training_data(train_dict_3d, opt)
    else:
        # raw training data from Human 3.6M (S15678)
        train_dict_3d = np.load(dict_path, allow_pickle=True).item()
    return train_dict_3d

def get_test_dict_3d(opt):
    """
    Get the testing 3d skeletons as a Python dictionary.

    Args
      opt: experiment options
    Returns
      test_dict_3d: a dictionary containing testing 3d poses
    """
    if opt.test_source == 'h36m':
        # for h36m
        dict_path = os.path.join(opt.data_dir, 'threeDPose_test.npy')
        test_dict_3d  = np.load(dict_path, allow_pickle=True).item()
    else:
        raise NotImplementedError
    return test_dict_3d
import logging,os,sys,time
def logger_print(epoch,
                 batch_idx,
                 loss
                 ):
    """
    Log training history.
    """
    msg = 'Train Epoch: {} [ ({:.0f}%)]\tLoss: {:.6f} '.format(
        epoch,
        batch_idx ,
        loss.data.item())
    logging.info(msg)
    return
def get_keys(data):
    data1,data5,data6,data7,data8= [], [], [], [], []
    for key in data.keys():
        sub, act, name = key
        # keytwo = train_dict_3d[key]
        if sub == 1:
            data1.append(data[key])
        if sub == 5:
            data5.append(data[key])
        if sub == 6:
            data6.append(data[key])
        if sub == 7:
            data7.append(data[key])
        if sub == 8:
            data8.append(data[key])
    return data1,data5,data6,data7,data8
# def get2ddata(mymodel, optim,data_train,data_targ,threeDcameras,opt,j,epoch,key):
#     data_train = data_train
#     data_targ = data_targ
#     threeDcameras = threeDcameras
#     data_dir = opt.data_dir
#     cameras_path = os.path.join(data_dir, 'cameras.npy')
#     rcams = np.load(cameras_path, allow_pickle=True).item()
#     R, T, f, c, k, p, name = rcams[(key[0], j % 4 + 1)]
#     j = j + 1
#     if opt.cuda:
#         with torch.no_grad():
#             # move to GPU
#             data_train = torch.Tensor(data_train)
#             data_targ = torch.Tensor(data_targ)
#             threeDcameras = torch.Tensor(threeDcameras)
#             data_train, data_targ, threeDcameras = data_train.cuda(), data_targ.cuda(), threeDcameras.cuda()
#
#     data_train = np.concatenate(data_train, axis=0)
#     data_targ = np.concatenate(data_targ, axis=0)
#     threeDcameras = np.concatenate(threeDcameras, axis=0)
#     optim = torch.optim.Adam(mymodel.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
#     optim.zero_grad()
#     # forward pass to get prediction
#     prediction = model(data_train)
#     # compute loss
#     # loss = 0.4*loss1 + 0.6*loss2
#     # smoothed l1 loss function
#     loss = F.smooth_l1_loss(prediction, data_targ)
#     loss.backward()
#     optim.step()
#     # logging
#     if j % 100 == 0:
#         logger_print(epoch,
#                      j,
#                      loss)

def adjust_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

import random
from torch.utils.data import  Dataset,DataLoader,TensorDataset
class trainset(Dataset):
    def __init__(self,target,data):
        #定义好 image 的路径
        self.target = target
        self.data = data
    def __getitem__(self, index):
        data = self.data[index]
        target = self.target[index]
        return target,data

    def __len__(self):
        return self.target.shape[0]
class trainset2(Dataset):
    def __init__(self,target,data,threeDcameras,f,c):
        #定义好 image 的路径
        self.target = target
        self.data = data
        self.threeDcameras=threeDcameras
        self.f=f
        self.c=c
    def __getitem__(self, index):
        data = self.data[index]
        target = self.target[index]
        threeDcameras=self.threeDcameras[index]
        f = self.f[index]
        c =self.c[index]
        return target,data,threeDcameras,f,c

    def __len__(self):
        return self.target.shape[0]
import random
def random_dic(dicts):
    dict_key_ls = list(dicts.keys())
    random.shuffle(dict_key_ls)
    new_dic = {}
    for key in dict_key_ls:
        new_dic[key] = dicts.get(key)
    return new_dic

def myShuffle(x, y, z, int=int):
    """x, random=random.random -> shuffle list x in place; return None.

    Optional arg random is a 0-argument function returning a random
    float in [0.0, 1.0); by default, the standard random.random.
    """

    # 转成numpy
    if torch.is_tensor(x) == True:
        x = x.numpy()
    if torch.is_tensor(y) == True:
        y = y.numpy()
    if torch.is_tensor(z) == True:
        z = z.numpy()
    random = np.random.random()
    # 开始随机置换
    for i in range(len(x)):
        j = int(random * (i + 1))
        if j <= len(x) - 1:  # 交换
            x[i], x[j] = x[j], x[i]
            y[i], y[j] = y[j], y[i]
            z[i], z[j] = z[j], z[i]
    # 转回tensor
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    z = torch.from_numpy(z)
    return x, y, z


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def get_dict_2d(train_dict_3d, test_dict_3d, rcams, ncams, opt):
    train_dict_2d = np.load(os.path.join(opt.data_dir, 'twoDPose_HRN_train.npy'), allow_pickle=True).item()
    test_dict_2d = np.load(os.path.join(opt.data_dir, 'twoDPose_HRN_test.npy'), allow_pickle=True).item()
    #
    # train_dict_2d_targ = project_to_cameras2(train_dict_3d, rcams, ncams=ncams)
    # test_dict_2d_train = {}
    # test_dict_2d_targ = project_to_cameras2(test_dict_3d, rcams, ncams=ncams)
    # train_dict_2d_train = {}
    # tran_all = []
    # target_all = []
    # threeDcameras_all=[]
    # f_all=[]
    # c_all=[]
    # train_dict_3d_convert = transform_world_to_camera(train_dict_3d, rcams, ncams=ncams)
    # test_dict_3d_convert = transform_world_to_camera(test_dict_3d, rcams, ncams=ncams)
    # j = 0
    # for key1, key2 in zip(sorted(train_dict_2d.keys()), sorted(train_dict_2d_targ.keys())):
    #     tran_all.append(train_dict_2d[key1])
    #     target_all.append(train_dict_2d_targ[key2])
    #     threeDcameras_all.append(train_dict_3d_convert[key2])
    #     R, T, f, c, k, p, name = rcams[(key2[0], j % 4 + 1)]
    #     j = j + 1
    #     for i in range(train_dict_2d_targ[key2].shape[0]):
    #         f_all.append(f)
    #         c_all.append(c)
    # tran_all = np.concatenate(tran_all, axis=0)
    # target_all = np.concatenate(target_all, axis=0)
    # threeDcameras_all = np.concatenate(threeDcameras_all, axis=0)
    # tran_all = np.normalize_axis_index(tran_all,axis=1)
    # target_all = np.normalize_axis_index(target_all,axis=1)
    # threeDcameras_all = np.normalize_axis_index(target_all,axis=1)
    # f_all = np.normalize_axis_index(target_all,axis=1)
    # c_all = np.normalize_axis_index(target_all,axis=1)
    # deal_dataset = trainset(target=target_all, data=tran_all)
    # dataloader = DataLoader(dataset=deal_dataset, batch_size=4096, shuffle=True)
    # deal_dataset2 = trainset2(target=target_all, data=tran_all,threeDcameras=threeDcameras_all,f=f_all,c=c_all)
    # dataloader2 = DataLoader(dataset=deal_dataset2, batch_size=1024, shuffle=True)
    #
    # for key in train_dict_2d_targ.keys():
    #     keynew = []
    #     keynew.append(key[0])
    #     keynew.append(key[1])
    #     keynew.append(key[2] + "-sh")
    #     keynew = tuple(keynew)
    #     train_dict_2d_train[key] = train_dict_2d[keynew]
    #
    #
    # for key in test_dict_2d_targ.keys():
    #     keynew = []
    #     keynew.append(key[0])
    #     keynew.append(key[1])
    #     keynew.append(key[2] + "-sh")
    #     keynew = tuple(keynew)
    #     if keynew in test_dict_2d:
    #         test_dict_2d_train[key] = test_dict_2d[keynew]
    #
    #
    #     # apply 3d post-processing (centering around root)
    # # train_dict_3d_convert, train_root_positions_convert = postprocess_3d(train_dict_3d_convert)
    # # onedata_train, fivedata_train, sixdata_train, sevendata_train, eightdata_train=get_keys(train_dict_2d_train)#训练出来的2D
    # # onedata_trag, fivedata_trag, sixdata_trag, sevendata_trag, eightdata_trag = get_keys(train_dict_2d_targ)#真实的2D
    # # onedata_cameras, fivedata_cameras, sixdata_cameras, sevendata_cameras, eightdata_cameras =get_keys(train_dict_3d_convert)#实际的3D
    # acts=[1,5,6,7,8]
    # opt = parse.parse_arg()
    # opt.cuda = opt.gpuid >= 0
    # mymodel = model.get_model(
    #                           refine_3d=opt.refine_3d,
    #                           norm_twoD=opt.norm_twoD,
    #                           num_blocks=opt.num_blocks,
    #                           input_size=64,
    #                           output_size=64,
    #                           linear_size=opt.linear_size,
    #                           dropout=opt.dropout,
    #                           leaky=opt.leaky)
    # train_dict_2d_pred = {}
    # test_dict_2d_pred = {}
    # # mymodel = torch.load('/media/zlz422/82BEDFE2BEDFCD33/jyt/project/EvSkeleton/data/human3.6M/2021-11-03 17:28:40.732213_jyt_fine_2d_no/model')
    # # mymodel.train()
    # # mymodel = torch.load(
    # #     '/media/zlz422/82BEDFE2BEDFCD33/jyt/project/EvSkeleton/data/human3.6M/2021-10-31 21:34:08.512679_jyt_fine_2d_zor/model')
    # mymodel = mymodel.cuda()
    # optim = torch.optim.Adam(mymodel.parameters(), lr=0.001, weight_decay=0.0)
    # # # optim = torch.optim.SGD(mymodel.parameters(),lr=0.001,momentum=opt.momentum,weight_decay=opt.weight_decay)
    # epoch = 20
    # for batch_idx in range(1, epoch + 1):
    #     # if batch_idx==20:
    #     #     adjust_lr(optim, 0.001)
    #     print("lr:{}".format(optim.state_dict()['param_groups'][0]['lr']))
    #     lossall=0
    #     # scheduler.step()
    #     for i, (data_targ, data_train,threeDcameras,f,c) in enumerate(dataloader2):
    #
    #         # data_targ = np.float(data_targ)
    #         # data_train = torch.Tensor(data_train)
    #         # data_targ = torch.Tensor(data_targ)
    #         data_targ = torch.tensor(data_targ, dtype=torch.float32)
    #         f =torch.tensor(f, dtype=torch.float32)
    #         c = torch.tensor(c, dtype=torch.float32)
    #         threeDcameras = torch.tensor(threeDcameras, dtype=torch.float32)
    #         if opt.cuda:
    #             with torch.no_grad():
    #                 # move to GPU
    #                 data_train = data_train.cuda()
    #                 data_targ = data_targ.cuda()
    #                 threeDcameras = threeDcameras.cuda()
    #                 f=f.cuda()
    #                 c=c.cuda()
    #
    #         optim.zero_grad()
    #         # forward pass to get prediction
    #         prediction = mymodel(data_train)
    #         threeDcameras_cum = threeDcameras.reshape(-1, 32, 3)
    #         prediction_cum = prediction.reshape(-1, 32, 2)
    #         target_cum = data_targ.reshape(-1, 32, 2)
    #         XdivZ = threeDcameras_cum[:, :, 0] / threeDcameras_cum[:, :, 2]
    #         YdivZ = threeDcameras_cum[:, :, 1] / threeDcameras_cum[:, :, 2]
    #         XX = torch.stack((XdivZ, YdivZ), 2)
    #         XX = XX.cuda()
    #         # fz=(torch.sum(XX * XX, axis=0) - n_smaples * (torch.mean(XX, axis=0)*torch.mean(XX, axis=0)))
    #         # fm=(torch.sum(target_cum * XX,axis=0) - n_smaples *(torch.mean(target_cum, axis=0) * torch.mean(XX, axis=0)))
    #         # fpred_target = fm/ fz
    #         # cpred_target = torch.mean(target_cum,dim=0)- torch.mean(XX,dim=0)*fpred_target
    #         # fpred_target = torch.mean(fpred_target,dim=0)
    #         # cpred_target = torch.mean(cpred_target,dim=0)
    #         fz = (torch.sum(XX * XX, axis=1) - 32 * (torch.mean(XX, axis=1) * torch.mean(XX, axis=1)))
    #         fm = (torch.sum(prediction_cum * XX, axis=1) - 32 * (torch.mean(prediction_cum, axis=1) * torch.mean(XX, axis=1)))
    #         fpred = fm / fz
    #         cpred = torch.mean(prediction_cum, dim=1) - torch.mean(XX, dim=1) * fpred
    #         # fpred = torch.mean(fpred, dim=0)
    #         # cpred = torch.mean(cpred, dim=0)
    #         fpred = fpred.cuda()
    #         cpred = cpred.cuda()
    #         f = f.squeeze()
    #         c = c.squeeze()
    #         loss1 = F.mse_loss(f, fpred)
    #         loss2 = F.mse_loss(c, cpred)
    #         loss3 = F.mse_loss(prediction, data_targ)
    #         # loss = F.mse_loss(prediction, data_targ)
    #         loss = 1*loss2 +  1*loss1 + 9*loss3
    #         # lossall=lossall+loss
    #         loss.backward()
    #         optim.step()
    #     print("Epoch:{} Loss：{} loss3:{} ".format(batch_idx,loss,loss3))


    # epoch = 20
    # adjust_lr(optim, 0.00001)
    # j = 0
    # epoch2 =1
    # # for w in range(3):
    # #     j = 0
    # #     if w==2:
    # #         adjust_lr(optim, 0.005)
    # for batch_idx in range(0, epoch):
    #     m = 0
    #     print("batch_idx:{}***********************************************************************".format(batch_idx))
    #     j = 0
    #     # if batch_idx % 2 == 0:
    #     #     keys = sorted(train_dict_2d_train.keys(), reverse=True)
    #     # else:
    #     #     keys = sorted(train_dict_2d_train.keys(), reverse=False)
    #
    #     for key in train_dict_2d_train.keys():
    #         for batch_idx2 in range(0, epoch2):
    #
    #         # print("lr:{}".format(optim.state_dict()['param_groups'][0]['lr']))
    #
    #             data_train = train_dict_2d_train[key]
    #             data_targ = train_dict_2d_targ[key]
    #             threeDcameras = train_dict_3d_convert[key]
    #             # get2ddata(mymodel, optim,data_train,data_targ,threeDcameras,opt,j,epoch,key)
    #
    #             data_dir = opt.data_dir
    #             cameras_path = os.path.join(data_dir, 'cameras.npy')
    #             rcams = np.load(cameras_path, allow_pickle=True).item()
    #             R, T, f, c, k, p, name = rcams[(key[0], j % 4 + 1)]
    #
    #             # data_train = np.concatenate(data_train, axis=0)
    #             # data_targ = np.concatenate(data_targ, axis=0)
    #             # threeDcameras = np.concatenate(threeDcameras, axis=0)
    #             data_train = torch.Tensor(data_train)
    #             data_targ = torch.Tensor(data_targ)
    #             threeDcameras = torch.Tensor(threeDcameras)
    #             # data_train, data_targ, threeDcameras = myShuffle(data_train, data_targ, threeDcameras)
    #             if opt.cuda:
    #                 with torch.no_grad():
    #                     # move to GPU
    #                     data_train = torch.Tensor(data_train)
    #                     data_targ = torch.Tensor(data_targ)
    #                     threeDcameras = torch.Tensor(threeDcameras)
    #                     data_train, data_targ, threeDcameras = data_train.cuda(), data_targ.cuda(), threeDcameras.cuda()
    #             optim.zero_grad()
    #             # forward pass to get prediction
    #             prediction = mymodel(data_train)
    #             # compute loss
    #             # loss = 0.4*loss1 + 0.6*loss2
    #             # smoothed l1 loss function
    #             n_smaples=threeDcameras.shape[0]
    #             threeDcameras_cum=threeDcameras.reshape(-1,32,3)
    #             prediction_cum = prediction.reshape(-1,32,2)
    #             target_cum = data_targ.reshape(-1, 32, 2)
    #             XdivZ = threeDcameras_cum[:,:, 0] / threeDcameras_cum[:,:, 2]
    #             YdivZ = threeDcameras_cum[:, :, 1] / threeDcameras_cum[:, :, 2]
    #             XX=torch.stack((XdivZ, YdivZ), 2)
    #             XX = XX.cuda()
    #             fz=(torch.sum(XX * XX, axis=0) - n_smaples * (torch.mean(XX, axis=0)*torch.mean(XX, axis=0)))
    #             fm=(torch.sum(target_cum * XX,axis=0) - n_smaples *(torch.mean(target_cum, axis=0) * torch.mean(XX, axis=0)))
    #             fpred_target = fm/ fz
    #             cpred_target = torch.mean(target_cum,dim=0)- torch.mean(XX,dim=0)*fpred_target
    #             fpred_target = torch.mean(fpred_target,dim=0)
    #             cpred_target = torch.mean(cpred_target,dim=0)
    #             fz = (torch.sum(XX * XX, axis=0) - n_smaples * (torch.mean(XX, axis=0) * torch.mean(XX, axis=0)))
    #             fm = (torch.sum(prediction_cum * XX, axis=0) - n_smaples * (torch.mean(prediction_cum, axis=0) * torch.mean(XX, axis=0)))
    #             fpred=fm/fz
    #             cpred = torch.mean(prediction_cum, dim=0) - torch.mean(XX, dim=0) * fpred
    #             fpred = torch.mean(fpred, dim=0)
    #             cpred = torch.mean(cpred, dim=0)
    #             fpred = fpred.cuda()
    #             cpred = cpred.cuda()
    #             f = torch.Tensor(f)
    #             c = torch.Tensor(c)
    #             f = f.squeeze()
    #             c = c.squeeze()
    #             f = f.cuda()
    #             c = c.cuda()
    #             loss1=F.mse_loss(f, fpred)
    #             loss2 = F.mse_loss(c, cpred)
    #             loss3 = F.mse_loss(prediction, data_targ)
    #             # loss = F.mse_loss(prediction, data_targ)
    #             loss=0.1*loss2+0.1*loss1+0.8*loss3
    #             # lossall=lossall+loss
    #             loss.backward()
    #             optim.step()
    #
    #             # logging
    #             # if batch_idx % 30 == 0:
    #         if m % 50 == 0:
    #             now = datetime.datetime.now()
    #             # print("Train target: f:{} c:{}   real target: f:{} c:{}".format(fpred_target, cpred_target, f, c))
    #             print("[{}] key:{} Train Epoch: {} Loss:  {} Loss3: {}".format(now, key,batch_idx, loss,loss3))
    #         m = m+1
    #         print("[{}] key:{} Train Epoch: {} Loss:  {} Loss3: {}".format(now, key, batch_idx, loss, loss3))
    #                 # print("[{}] key:{} Train Epoch: {} Loss:  {}".format(now, key, batch_idx, loss))
    #                 # print("[{}] key:{} f: {} c {}  prdef :{} predc :{} Train Epoch: {} Loss:{:.3f}".format(now,key,f.detach().cpu().numpy(),c.detach().cpu().numpy(),fpred.detach().cpu().numpy(),cpred.detach().cpu().numpy(),batch_idx,loss))
    #         j = j + 1
    #     if j%200==0:
    #         now = datetime.datetime.now()
    #         stpath = str(now) + "_" + "jyt_fine_2d_mid"
    #         save2Dpath = os.path.join(opt.data_dir, stpath)
    #         if not os.path.exists(save2Dpath):
    #             os.mkdir(save2Dpath)
    #         save2Dpath = save2Dpath + "/" + "model"
    #         torch.save(mymodel, save2Dpath)
    #         print("save中间model")
    #
    # epoch =20
    # adjust_lr(optim, 0.0001)
    # for batch_idx in range(1, epoch + 1):
    #     print("lr:{}".format(optim.state_dict()['param_groups'][0]['lr']))
    #     lossall = 0
    #     # scheduler.step()
    #     for i, (data_targ, data_train) in enumerate(dataloader):
    #
    #         # data_targ = np.float(data_targ)
    #         # data_train = torch.Tensor(data_train)
    #         # data_targ = torch.Tensor(data_targ)
    #         data_targ = torch.tensor(data_targ, dtype=torch.float32)
    #         if opt.cuda:
    #             with torch.no_grad():
    #                 # move to GPU
    #                 data_train = data_train.cuda()
    #                 data_targ = data_targ.cuda()
    #         optim.zero_grad()
    #         # forward pass to get prediction
    #         prediction = mymodel(data_train)
    #         loss = F.mse_loss(prediction, data_targ)
    #         loss.backward()
    #         lossall += loss
    #         optim.step()
    #     print("Epoch:{} Loss：{}".format(batch_idx, loss))
    # now = datetime.datetime.now()
    # stpath = str(now) + "_" + "jyt_fine_2d_no"
    # save2Dpath = os.path.join(opt.data_dir, stpath)
    # if not os.path.exists(save2Dpath):
    #     os.mkdir(save2Dpath)
    # save2Dpath = save2Dpath + "/" + "model"
    # torch.save(mymodel, save2Dpath)
    #
    # # mymodel.eval()
    # for key in test_dict_2d.keys():
    #     # for i in range(50):
    #     # mymodel=mymodel.cuda()
    #     data_train = test_dict_2d[key]
    #     data_train = torch.Tensor(data_train)
    #     data_train = data_train.cuda()
    #     # mymodel = mymodel.cpu()
    #     prediction = mymodel(data_train)
    #     keynew = []
    #     keynew.append(key[0])
    #     keynew.append(key[1])
    #     key2 = key[2].replace("-sh", "")
    #     keynew.append(key2)
    #     keynew = tuple(keynew)
    #     data_target = test_dict_2d_targ[keynew]
    #     # print("Key: {} predictionshape:{} targtshape:{} ".format(key, prediction.shape, data_target.shape))
    #     data_target = torch.Tensor(data_target)
    #     data_target = data_target.cuda()
    #     loss = F.mse_loss(prediction, data_target)
    #     # loss.backward()
    #     # if i % 5 == 0:
    #     #     print("TestLoss: {}".format(loss))
    #     # if i==49:
    #     #     loss2 = F.mse_loss(prediction, data_target)
    #     print("key:{}   TTrainLoss2: {}".format(key,loss))
    #     prediction = prediction.detach().cpu().numpy()
    #
    #         # prediction = [ predictionitem.detach().numpy() for predictionitem in prediction]
    #     test_dict_2d_pred[key] = prediction
    # for key in train_dict_2d.keys():
    #     #         # for i in range(50):
    #     mymodel2 = mymodel.cuda()
    #     data_train = train_dict_2d[key]
    #     data_train = torch.Tensor(data_train)
    #     data_train = data_train.cuda()
    #     # mymodel = mymodel.cpu()
    #     prediction = mymodel2(data_train)
    #     keynew = []
    #     keynew.append(key[0])
    #     keynew.append(key[1])
    #     key2 = key[2].replace("-sh", "")
    #     keynew.append(key2)
    #     keynew = tuple(keynew)
    #     data_target = train_dict_2d_targ[keynew]
    #     # print("Key: {} datshape:{} targtshape:{} ".format(key, data_train.shape, data_target.shape))
    #     data_target = torch.Tensor(data_target)
    #     data_target = data_target.cuda()
    #     loss = F.mse_loss(prediction, data_target)
    #     # if i%5==0:
    #     #     print("TrainLoss: {}".format(loss))
    #     # loss.backward()
    #     # if i == 49:
    #     #     loss2 = F.mse_loss(prediction, data_target)
    #     print("key:{}   FTrainLoss: {} ".format(key, loss))
    #     prediction = prediction.detach().cpu().numpy()
    #     # prediction = [ predictionitem.detach().numpy() for predictionitem in prediction]
    #     train_dict_2d_pred[key] = prediction

    # save_path = opt.data_dir+str(now)+"2drefinetrain"
    # np.save(save_path,train_dict_2d_pred)
    # save_path = opt.data_dir+str(now) + "2drefinetest"
    # np.save(save_path, test_dict_2d_pred)
    # train_dict_2d_pred = np.load(os.path.join(opt.data_dir, '2drefinetrain.npy'), allow_pickle=True).item()
    # test_dict_2d_pred = np.load(os.path.join(opt.data_dir, '2drefinetest.npy'), allow_pickle=True).item()
    return train_dict_2d, test_dict_2d, train_dict_3d
from numpy.core.multiarray import normalize_axis_index
def dimUse(dim, predict_14=False,
                        norm_twoD=True,
                        use_nose=True):
    if dim == 2:
        if not use_nose:
            dimensions_to_use = np.where(np.array([x != '' and x != 'Neck/Nose' for x in H36M_NAMES]))[0]
        else:
            dimensions_to_use = np.where(np.array([x != '' for x in H36M_NAMES]))[0]
        if norm_twoD:
            dimensions_to_use = np.delete(dimensions_to_use, 0)
        dimensions_to_use = np.sort(np.hstack((dimensions_to_use * 2,
                                               dimensions_to_use * 2 + 1)))
    else:
        dimensions_to_use = np.where(np.array([x != '' for x in H36M_NAMES]))[0]
        # hip is deleted
        # spine and neck are also deleted if predict_14
        dimensions_to_use = np.delete(dimensions_to_use, [0, 7, 9] if predict_14 else 0)
        dimensions_to_use = np.sort(np.hstack((dimensions_to_use * 3,
                                               dimensions_to_use * 3 + 1,
                                               dimensions_to_use * 3 + 2)))
        dimensions_to_ignore = np.delete(np.arange(len(H36M_NAMES) * 3),
                                         dimensions_to_use)


    return  dimensions_to_use
class DataPrefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()

        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.data_targ, self.data_train, self.threeDcameras, self.f, self.c= next(self.loader)
        except StopIteration:
            self.data_targ = None
            self.data_train = None
            self.threeDcameras = None
            self.f = None
            self.c = None
            return
        with torch.cuda.stream(self.stream):
            self.data_targ = self.data_targ.cuda(non_blocking=True)
            self.data_train = self.data_train.cuda(non_blocking=True)
            self.threeDcameras = self.threeDcameras.cuda(non_blocking=True)
            self.f = self.f.cuda(non_blocking=True)
            self.c = self.c.cuda(non_blocking=True)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()


    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data_targ = self.data_targ
        data_train = self.data_train
        threeDcameras = self.threeDcameras
        f = self.f
        c = self.c
        self.preload()
        return  data_targ,data_train,threeDcameras,f,c
def prepare_data_dict(rcams,
                      opt,
                      ncams=4,
                      predict_14=False,
                      use_nose=True
                      ):
    """
    Prepare 2D and 3D data as Python dictionaries.

    Args
      rcams: camera parameters
      opt: experiment options
      ncams: number of camera to use
      predict_14: whether to predict 14 joints or not
      use_nose: whether to use nose joint or not
    Returns
      data_dic: a dictionary containing training and testing data
      data_stats: statistics computed from training data
    """
    assert opt.twoD_source in ['synthetic', 'HRN'], 'Unknown 2D key-point type.'
    data_dic = {}
    # get 3D skeleton data
    train_dict_3d = get_train_dict_3d(opt)
    test_dict_3d = get_test_dict_3d(opt)
    # get 2D key-point data
    train_dict_2d, test_dict_2d, train_dict_3d2 = get_dict_2d(train_dict_3d,
                                                             test_dict_3d,
                                                             rcams,
                                                             ncams,
                                                             opt
                                                             )
    # compute normalization statistics and normalize the 2D data
    if opt.train:
        complete_train_2d = copy.deepcopy(np.vstack(list(train_dict_2d.values())))
        data_mean_2d, data_std_2d, dim_to_ignore_2d, dim_to_use_2d = \
            normalization_stats(complete_train_2d,
                                dim=2,
                                norm_twoD=True,
                                use_nose=use_nose
                                )

        train_dict_2d = normalize_data(train_dict_2d,
                                                  data_mean_2d,
                                                  data_std_2d,
                                                  dim_to_use_2d,
                                                  norm_single=opt.norm_single
                                                  )
    else:
        _, data_stats = load_ckpt(opt)
        data_mean_2d, data_std_2d = data_stats['mean_2d'], data_stats['std_2d']
        dim_to_use_2d = data_stats['dim_use_2d']
    complete_test_2d = copy.deepcopy(np.vstack(list(test_dict_2d.values())))
    data_mean_2d_test, data_std_2d_test, dim_to_ignore_2d_test, dim_to_use_2d_test = \
        normalization_stats(complete_test_2d,
                            dim=2,
                            norm_twoD=opt.norm_twoD,
                            use_nose=use_nose
                            )
    test_dict_2d = normalize_data(test_dict_2d,
                                             data_mean_2d_test,
                                             data_std_2d_test,
                                             dim_to_use_2d_test,
                                             norm_single=opt.norm_single
                                             )
    train_dict_2d_targ = project_to_cameras2(train_dict_3d, rcams, ncams=ncams)
    test_dict_2d_targ = project_to_cameras2(test_dict_3d, rcams, ncams=ncams)
    complete_train_dict_2d_targ_2d = copy.deepcopy(np.vstack(list(train_dict_2d_targ.values())))
    targ_data_mean_2d, targ_data_std_2d = \
        normalization_stats2(complete_train_dict_2d_targ_2d,
                             dim=2,
                             norm_twoD=opt.norm_twoD,
                             use_nose=use_nose
                             )
    train_dict_2d_targ = normalize_data2(train_dict_2d_targ,
                                   targ_data_mean_2d,
                                   targ_data_std_2d,
                                   dim_to_use_2d,
                                   norm_single=opt.norm_single
                                   )
    complete_test_2d = copy.deepcopy(np.vstack(list(test_dict_2d_targ.values())))
    targ_test_data_mean_2d, targ_test_data_std_2d = \
        normalization_stats2(complete_test_2d,
                             dim=2,
                             norm_twoD=opt.norm_twoD,
                             use_nose=use_nose
                             )
    test_dict_2d_targ = normalize_data2(test_dict_2d_targ,
                                         targ_test_data_std_2d,
                                         targ_data_std_2d,
                                         dim_to_use_2d,
                                         norm_single=opt.norm_single
                                         )
    tran_all = []
    target_all = []
    threeDcameras_all=[]
    f_all=[]
    c_all=[]
    dimUsethree = dimUse(3)
    train_dict_3d_convert = transform_world_to_camera(train_dict_3d, rcams, ncams=ncams)
    train_dict_3d, test_root_positions = postprocess_3d(train_dict_3d)

    for key in train_dict_3d_convert.keys():
        train_dict_3d_convert[key]=train_dict_3d_convert[key][:, dimUsethree]
    complete_data = copy.deepcopy(np.vstack(list(train_dict_3d_convert.values())))
    data_mean = np.mean(complete_data, axis=0)
    data_std = np.std(complete_data, axis=0)
    for key in train_dict_3d_convert.keys():
        mu = data_mean
        stddev = data_std
        train_dict_3d_convert[key] = np.divide((train_dict_3d_convert[key] - mu),
                                                    stddev)
    j = 0

    for key1, key2 in zip(sorted(train_dict_2d.keys()), sorted(train_dict_2d_targ.keys())):
        tran_all.append(train_dict_2d[key1])
        target_all.append(train_dict_2d_targ[key2])
        threeDcameras_all.append(train_dict_3d_convert[key2])
        R, T, f, c, k, p, name = rcams[(key2[0], j % 4 + 1)]
        j = j + 1
        for i in range(train_dict_2d_targ[key2].shape[0]):
            f_all.append(f)
            c_all.append(c)
    tran_all = np.concatenate(tran_all, axis=0)
    target_all = np.concatenate(target_all, axis=0)
    threeDcameras_all = np.concatenate(threeDcameras_all, axis=0)

    f_all_data_mean_2d, f_all_data_std_2d = \
        normalization_stats2(f_all,
                             dim=2,
                             norm_twoD=opt.norm_twoD,
                             use_nose=use_nose
                             )
    f_all = normalize_data3(f_all,
                                         f_all_data_mean_2d,
                                         f_all_data_std_2d,
                                         norm_single=opt.norm_single
                                         )
    c_all_data_mean_2d, c_all_data_std_2d = \
        normalization_stats2(c_all,
                             dim=2,
                             norm_twoD=opt.norm_twoD,
                             use_nose=use_nose
                             )
    c_all = normalize_data3(c_all,
                            c_all_data_mean_2d,
                            c_all_data_std_2d,
                            norm_single=opt.norm_single
                            )
    deal_dataset2 = trainset2(target=target_all, data=tran_all,threeDcameras=threeDcameras_all,f=f_all,c=c_all)
    dataloader2 = DataLoader(dataset=deal_dataset2, batch_size=256, shuffle=True)
    opt = parse.parse_arg()
    opt.cuda = opt.gpuid >= 0
    mymodel = model.get_model(
                              refine_3d=opt.refine_3d,
                              norm_twoD=opt.norm_twoD,
                              num_blocks=opt.num_blocks,
                              input_size=32,
                              output_size=32,
                              linear_size=opt.linear_size,
                              dropout=opt.dropout,
                              leaky=opt.leaky)
    mymodel = mymodel.cuda()
    optim = torch.optim.Adam(mymodel.parameters(), lr=0.001, weight_decay= 0.0001)
    # # optim = torch.optim.SGD(mymodel.parameters(),lr=0.001,momentum=opt.momentum,weight_decay=opt.weight_decay)
    epoch = 100
    for batch_idx in range(1, epoch + 1):
        if batch_idx == 40:
            adjust_lr(optim, 0.0005)
        if batch_idx == 50:
            adjust_lr(optim, 0.0001)
        if batch_idx == 60:
            adjust_lr(optim, 0.00005)
        if batch_idx == 80:
            adjust_lr(optim, 0.00001)
        print("lr:{}".format(optim.state_dict()['param_groups'][0]['lr']))
        lossall=0
        # scheduler.step()
        prefetcher = DataPrefetcher(dataloader2)
        data_targ, data_train, threeDcameras, f, c = prefetcher.next()
        iteration = 0
        while data_targ is not None:
            iteration += 1
            # if iteration==10:
            #     adjust_lr(optim,0.0001)

            # data_targ = np.float(data_targ)
            # data_train = torch.Tensor(data_train)
            # data_targ = torch.Tensor(data_targ)
            data_targ = torch.tensor(data_targ, dtype=torch.float32)
            f =torch.tensor(f, dtype=torch.float32)
            c = torch.tensor(c, dtype=torch.float32)
            threeDcameras = torch.tensor(threeDcameras, dtype=torch.float32)
            if opt.cuda:
                with torch.no_grad():
                    # move to GPU
                    data_train = data_train.cuda()
                    data_targ = data_targ.cuda()
                    threeDcameras = threeDcameras.cuda()
                    f=f.cuda()
                    c=c.cuda()

            optim.zero_grad()
            # forward pass to get prediction
            prediction = mymodel(data_train)
            threeDcameras_cum = threeDcameras.reshape(-1, 16, 3)
            prediction_cum = prediction.reshape(-1, 16, 2)
            XdivZ = threeDcameras_cum[:, :, 0] / threeDcameras_cum[:, :, 2]
            YdivZ = threeDcameras_cum[:, :, 1] / threeDcameras_cum[:, :, 2]
            XX = torch.stack((XdivZ, YdivZ), 2)
            XX = XX.cuda()
            fz = (torch.sum(XX * XX, axis=1) - 16 * (torch.mean(XX, axis=1) * torch.mean(XX, axis=1)))
            fm = (torch.sum(prediction_cum * XX, axis=1) - 16 * (torch.mean(prediction_cum, axis=1) * torch.mean(XX, axis=1)))
            fpred = fm / fz
            cpred = torch.mean(prediction_cum, dim=1) - torch.mean(XX, dim=1) * fpred
            fpred = fpred.cuda()
            cpred = cpred.cuda()
            f = f.squeeze()
            c = c.squeeze()
            loss1 = F.mse_loss(f, fpred)
            loss2 = F.mse_loss(c, cpred)
            loss3 = F.mse_loss(prediction, data_targ)
            # loss = F.mse_loss(prediction, data_targ)
            loss = 0.1*loss2 +  0.1*loss1 + 0.9*loss3
            lossall=lossall+loss
            loss.backward()
            optim.step()
            data_targ, data_train, threeDcameras, f, c = prefetcher.next()
        print("Epoch:{} LossAll:{}  Loss：{} loss3:{} ".format(batch_idx,lossall/iteration,loss,loss3))

    test_dict_2d_pred={}
    train_dict_2d_pred = {}
        # mymodel.eval()
    for key in test_dict_2d.keys():
        data_train = test_dict_2d[key]
        data_train = torch.Tensor(data_train)
        data_train = data_train.cuda()
        # mymodel = mymodel.cpu()
        prediction = mymodel(data_train)
        keynew = []
        keynew.append(key[0])
        keynew.append(key[1])
        key2 = key[2].replace("-sh", "")
        keynew.append(key2)
        keynew = tuple(keynew)
        data_target = test_dict_2d_targ[keynew]
        # print("Key: {} predictionshape:{} targtshape:{} ".format(key, prediction.shape, data_target.shape))
        data_target = torch.Tensor(data_target)
        data_target = data_target.cuda()
        loss = F.mse_loss(prediction, data_target)
        # loss.backward()
        # if i % 5 == 0:
        #     print("TestLoss: {}".format(loss))
        # if i==49:
        #     loss2 = F.mse_loss(prediction, data_target)
        print("key:{}   TTrainLoss2: {}".format(key, loss))
        prediction = prediction.detach().cpu().numpy()

        # prediction = [ predictionitem.detach().numpy() for predictionitem in prediction]
        test_dict_2d_pred[key] = prediction
    for key in train_dict_2d.keys():
        #         # for i in range(50):
        data_train = train_dict_2d[key]
        data_train = torch.Tensor(data_train)
        data_train = data_train.cuda()
        # mymodel = mymodel.cpu()
        prediction = mymodel(data_train)
        keynew = []
        keynew.append(key[0])
        keynew.append(key[1])
        key2 = key[2].replace("-sh", "")
        keynew.append(key2)
        keynew = tuple(keynew)
        data_target = train_dict_2d_targ[keynew]
        # print("Key: {} datshape:{} targtshape:{} ".format(key, data_train.shape, data_target.shape))
        data_target = torch.Tensor(data_target)
        data_target = data_target.cuda()
        loss = F.mse_loss(prediction, data_target)
        # if i%5==0:
        #     print("TrainLoss: {}".format(loss))
        # loss.backward()
        # if i == 49:
        #     loss2 = F.mse_loss(prediction, data_target)
        print("key:{}   FTrainLoss: {} ".format(key, loss))
        prediction = prediction.detach().cpu().numpy()
        # prediction = [ predictionitem.detach().numpy() for predictionitem in prediction]
        train_dict_2d_pred[key] = prediction
    # for key in train_dict_2d_pred.keys():
    #     train_dict_2d_pred[key]=train_dict_2d_pred[key]
    # for key in test_dict_2d_pred.keys():
    #     test_dict_2d_pred[key] = test_dict_2d_pred[key]
    data_dic['train_set_2d']=train_dict_2d_pred
    data_dic['test_set_2d'] = test_dict_2d_pred
    # The 3D joint position is represented in the world coordinate,
    # which is converted to camera coordinate system as the regression target
    if opt.train:
        train_dict_3d = transform_world_to_camera(train_dict_3d, rcams, ncams=ncams)
        # apply 3d post-processing (centering around root)
        train_dict_3d, train_root_positions = postprocess_3d(train_dict_3d)
    test_dict_3d  = transform_world_to_camera(test_dict_3d, rcams, ncams=ncams)
    test_dict_3d,  test_root_positions  = postprocess_3d(test_dict_3d)
    if opt.train:
        # compute normalization statistics and normalize the 3D data
        complete_train_3d = copy.deepcopy(np.vstack(list(train_dict_3d.values())))
        data_mean_3d, data_std_3d, dim_to_ignore_3d, dim_to_use_3d =\
        normalization_stats(complete_train_3d, dim=3, predict_14=predict_14)
        data_dic['train_set_3d'] = normalize_data(train_dict_3d,
                                                  data_mean_3d,
                                                  data_std_3d,
                                                  dim_to_use_3d
                                                  )
        # some joints are not used during training
        dim_use_2d = list_remove([i for i in range(len(data_mean_2d))],
                                  list(dim_to_ignore_2d))
        dim_use_3d = list_remove([i for i in range(len(data_mean_3d))],
                                  list(dim_to_ignore_3d))
        # assemble a dictionary for data statistics
        data_stats = {'mean_2d':data_mean_2d,
                      'std_2d':data_std_2d,
                      'mean_3d':data_mean_3d,
                      'std_3d':data_std_3d,
                      'dim_ignore_2d':dim_to_ignore_2d,
                      'dim_ignore_3d':dim_to_ignore_3d,
                      'dim_use_2d':dim_use_2d,
                      'dim_use_3d':dim_use_3d
                      }
    else:
        data_mean_3d, data_std_3d = data_stats['mean_3d'], data_stats['std_3d']
        dim_to_use_3d = data_stats['dim_use_3d']
    data_dic['test_set_3d']  = normalize_data(test_dict_3d,
                                              data_mean_3d,
                                              data_std_3d,
                                              dim_to_use_3d
                                              )

    return data_dic, data_stats

def select_action(dic_2d, dic_3d, action, twoD_source):
    """
    Construct sub-dictionaries by specifying which action to use

    Args
        dic_2d: dictionary containing 2d poses
        dic_3d: dictionary containing 3d poses
        action: the action to use
        twoD_source: how the key-points are generated (synthetic or real)
    Returns
        dic_2d_action: sub-dictionary containing 2d poses for the specified action
        dic_3d_action: sub-dictionary containing 3d poses for the specified action
    """
    dic_2d_action = {}
    dic_3d_action = {}
    for key in dic_2d.keys():
        if key[1] == action:
            dic_2d_action[key] = dic_2d[key].copy()
            if twoD_source == 'synthetic':
                key3d = key
            else:
                key3d = (key[0], key[1], key[2][:-3])
            dic_3d_action[key3d] = dic_3d[key3d].copy()
    return dic_2d_action, dic_3d_action

def split_action(dic_2d, dic_3d, actions, camera_frame, opt, input_size, output_size):
    """
    Generate a list of datasets for each action.

    Args
        dic_2d: dictionary containing 2d poses
        dic_3d: dictionary containing 3d poses
        actions: list of defined actions
        camera_frame: use camera coordinate system
        opt: experiment options
        input_size: input vector length
        output_size: output vector length
    Returns
        action_dataset_list: a list of datasets where each element correspond
        to one action
    """
    action_dataset_list = []
    for act_id in range(len(actions)):
        action = actions[act_id]
        dic_2d_action, dic_3d_action = select_action(dic_2d, dic_3d, action, opt.twoD_source)
        eval_input, eval_output = get_all_data(dic_2d_action,
                                               dic_3d_action,
                                               camera_frame,
                                               norm_twoD=opt.norm_twoD,
                                               input_size=input_size,
                                               output_size=output_size)
        action_dataset = dataset.PoseDataset(eval_input,
                                             eval_output,
                                             'eval',
                                             action_name=action,
                                             refine_3d=opt.refine_3d)
        action_dataset_list.append(action_dataset)
    return action_dataset_list
def normalization_stats2(complete_data,
                        dim,
                        predict_14=False,
                        norm_twoD=False,
                        use_nose=False
                        ):
    """
    Computes normalization statistics: mean and stdev, dimensions used and ignored

    Args
        complete_data: nxd np array with poses
        dim. integer={2,3} dimensionality of the data
        predict_14. boolean. Whether to use only 14 joints
        use_nose: whether to use nose or not
    Returns
        data_mean: np vector with the mean of the data
        data_std: np vector with the standard deviation of the data
        dimensions_to_ignore: list of dimensions not used in the model
        dimensions_to_use: list of dimensions used in the model
    """
    if not dim in [2,3]:
        raise(ValueError, 'dim must be 2 or 3')
    data_mean = np.mean(complete_data, axis=0)
    data_std  =  np.std(complete_data, axis=0)
    # Encodes which 17 (or 14) 2d-3d pairs we are predicting
    dimensions_to_ignore = []
    return data_mean, data_std
def normalization_stats(complete_data,
                        dim,
                        predict_14=False,
                        norm_twoD=False,
                        use_nose=False
                        ):
    """
    Computes normalization statistics: mean and stdev, dimensions used and ignored

    Args
        complete_data: nxd np array with poses
        dim. integer={2,3} dimensionality of the data
        predict_14. boolean. Whether to use only 14 joints
        use_nose: whether to use nose or not
    Returns
        data_mean: np vector with the mean of the data
        data_std: np vector with the standard deviation of the data
        dimensions_to_ignore: list of dimensions not used in the model
        dimensions_to_use: list of dimensions used in the model
    """
    if not dim in [2,3]:
        raise(ValueError, 'dim must be 2 or 3')
    data_mean = np.mean(complete_data, axis=0)
    data_std  =  np.std(complete_data, axis=0)
    # Encodes which 17 (or 14) 2d-3d pairs we are predicting
    dimensions_to_ignore = []
    if dim == 2:
        if not use_nose:
            dimensions_to_use = np.where(np.array([x != '' and x != 'Neck/Nose' for x in H36M_NAMES]))[0]
        else:
            dimensions_to_use = np.where(np.array([x != '' for x in H36M_NAMES]))[0]
        if norm_twoD:
            dimensions_to_use = np.delete(dimensions_to_use, 0)
        dimensions_to_use = np.sort(np.hstack((dimensions_to_use*2,
                                               dimensions_to_use*2+1)))
        dimensions_to_ignore = np.delete(np.arange(len(H36M_NAMES)*2),
                                         dimensions_to_use)
    else:
        dimensions_to_use = np.where(np.array([x != '' for x in H36M_NAMES]))[0]
        # hip is deleted
        # spine and neck are also deleted if predict_14
        dimensions_to_use = np.delete(dimensions_to_use, [0,7,9] if predict_14 else 0)
        dimensions_to_use = np.sort(np.hstack((dimensions_to_use*3,
                                               dimensions_to_use*3+1,
                                               dimensions_to_use*3+2)))
        dimensions_to_ignore = np.delete(np.arange(len(H36M_NAMES)*3),
                                         dimensions_to_use)
    return data_mean, data_std, dimensions_to_ignore, dimensions_to_use
def normalization_stats3(complete_data,
                        dim,
                        predict_14=False,
                        norm_twoD=False,
                        use_nose=False
                        ):
    """
    Computes normalization statistics: mean and stdev, dimensions used and ignored

    Args
        complete_data: nxd np array with poses
        dim. integer={2,3} dimensionality of the data
        predict_14. boolean. Whether to use only 14 joints
        use_nose: whether to use nose or not
    Returns
        data_mean: np vector with the mean of the data
        data_std: np vector with the standard deviation of the data
        dimensions_to_ignore: list of dimensions not used in the model
        dimensions_to_use: list of dimensions used in the model
    """
    if not dim in [2,3]:
        raise(ValueError, 'dim must be 2 or 3')
    if dim == 2:
        if not use_nose:
            dimensions_to_use = np.where(np.array([x != '' and x != 'Neck/Nose' for x in H36M_NAMES]))[0]
        else:
            dimensions_to_use = np.where(np.array([x != '' for x in H36M_NAMES]))[0]
        if norm_twoD:
            dimensions_to_use = np.delete(dimensions_to_use, 0)
        dimensions_to_use = np.sort(np.hstack((dimensions_to_use*2,
                                               dimensions_to_use*2+1)))
        dimensions_to_ignore = np.delete(np.arange(len(H36M_NAMES)*2),
                                         dimensions_to_use)
    else:
        dimensions_to_use = np.where(np.array([x != '' for x in H36M_NAMES]))[0]
        # hip is deleted
        # spine and neck are also deleted if predict_14
        dimensions_to_use = np.delete(dimensions_to_use, [0,7,9] if predict_14 else 0)
        dimensions_to_use = np.sort(np.hstack((dimensions_to_use*3,
                                               dimensions_to_use*3+1,
                                               dimensions_to_use*3+2)))
        dimensions_to_ignore = np.delete(np.arange(len(H36M_NAMES)*3),
                                         dimensions_to_use)
    return dimensions_to_ignore, dimensions_to_use
def transform_world_to_camera(poses_set, cams, ncams=4):
    """
    Transform 3d poses from world coordinate to camera coordinate system
    Args
      poses_set: dictionary with 3d poses
      cams: dictionary with cameras
      ncams: number of cameras per subject
    Return:
      t3d_camera: dictionary with 3d poses in camera coordinate
    """
    t3d_camera = {}
    for t3dk in sorted(poses_set.keys()):
      subj, action, seqname = t3dk
      t3d_world = poses_set[t3dk]
      for c in range(ncams):
        R, T, f, c, k, p, name = cams[(subj, c+1)]
        camera_coord = cameras.world_to_camera_frame(np.reshape(t3d_world, [-1, 3]), R, T)
        camera_coord = np.reshape(camera_coord, [-1, len(H36M_NAMES)*3])
        sname = seqname[:-3]+"."+name+".h5" # e.g.: Waiting 1.58860488.h5
        t3d_camera[(subj, action, sname)] = camera_coord
    return t3d_camera
def normalize_data2(data, data_mean, data_std, dim_to_use_2d,norm_single=False):
    """
    Normalizes a dictionary of poses

    Args
        data: dictionary where values are
        data_mean: np vector with the mean of the data
        data_std: np vector with the standard deviation of the data
        dim_to_use: list of dimensions to keep in the data
        norm_single: whether to perform normalization independently for each
        sample
    Returns
        data_out: dictionary with same keys as data, but values have been normalized
    """
    data_out = {}
    for key in data.keys():
        data[key] = data[key][:, dim_to_use_2d]
        if norm_single:
            # does not use statistics over the whole dataset
            temp = data[key]
            temp = temp.reshape(len(temp), -1, 2)
            mean_x = np.mean(temp[:,:,0], axis=1).reshape(len(temp), 1)
            std_x = np.std(temp[:,:,0], axis=1)
            mean_y = np.mean(temp[:,:,1], axis=1).reshape(len(temp), 1)
            std_y = np.std(temp[:,:,1], axis=1)
            denominator = (0.5*(std_x + std_y)).reshape(len(std_x), 1)
            temp[:,:,0] = (temp[:,:,0] - mean_x)/denominator
            temp[:,:,1] = (temp[:,:,1] - mean_y)/denominator
            data_out[key] = temp.reshape(len(temp), -1)
        else:
            mu = data_mean[dim_to_use_2d]
            stddev = data_std[dim_to_use_2d]
            # data_out[key]=data[key]
            data_out[ key ]= np.divide( (data[key]- mu), stddev)
    return data_out
def normalize_data4(data, data_mean, data_std, dim_to_use_2d,dim_to_ignore_2d,norm_single=False):
    """
    Normalizes a dictionary of poses

    Args
        data: dictionary where values are
        data_mean: np vector with the mean of the data
        data_std: np vector with the standard deviation of the data
        dim_to_use: list of dimensions to keep in the data
        norm_single: whether to perform normalization independently for each
        sample
    Returns
        data_out: dictionary with same keys as data, but values have been normalized
    """
    data_out = {}
    for key in data.keys():

        mu = data_mean
        stddev = data_std
        data_out[key]=data[key]
        data_out[key][:,dim_to_ignore_2d]=0
        data_out[ key ][:,dim_to_use_2d] = np.divide( (data[key][:,dim_to_use_2d] - mu[dim_to_use_2d]), stddev[dim_to_use_2d] )
    return data_out
def normalize_data3(data, data_mean, data_std,norm_single=False):

    mu = data_mean
    stddev = data_std

    data_out= np.divide( (data - mu), stddev)
    return data_out
def normalize_data(data, data_mean, data_std, dim_to_use, norm_single=False):
    """
    Normalizes a dictionary of poses

    Args
        data: dictionary where values are
        data_mean: np vector with the mean of the data
        data_std: np vector with the standard deviation of the data
        dim_to_use: list of dimensions to keep in the data
        norm_single: whether to perform normalization independently for each
        sample
    Returns
        data_out: dictionary with same keys as data, but values have been normalized
    """
    data_out = {}
    for key in data.keys():
        data[ key ] = data[ key ][ :, dim_to_use ]
        if norm_single:
            # does not use statistics over the whole dataset
            temp = data[key]
            temp = temp.reshape(len(temp), -1, 2)
            mean_x = np.mean(temp[:,:,0], axis=1).reshape(len(temp), 1)
            std_x = np.std(temp[:,:,0], axis=1)
            mean_y = np.mean(temp[:,:,1], axis=1).reshape(len(temp), 1)
            std_y = np.std(temp[:,:,1], axis=1)
            denominator = (0.5*(std_x + std_y)).reshape(len(std_x), 1)
            temp[:,:,0] = (temp[:,:,0] - mean_x)/denominator
            temp[:,:,1] = (temp[:,:,1] - mean_y)/denominator
            data_out[key] = temp.reshape(len(temp), -1)
        else:
            mu = data_mean[dim_to_use]
            stddev = data_std[dim_to_use]

            data_out[ key ] = np.divide( (data[key] - mu), stddev )
    return data_out

def unNormalizeData(normalized_data, data_mean, data_std, dimensions_to_ignore):
    """
    Un-normalizes a matrix whose mean has been substracted and that has been
    divided by standard deviation. Some dimensions might also be missing.

    Args
    normalized_data: nxd matrix to unnormalize
    data_mean: np vector with the mean of the data
    data_std: np vector with the standard deviation of the data
    dimensions_to_ignore: list of dimensions that were removed from the original data
    Returns
    orig_data: the unnormalized data
    """
    T = normalized_data.shape[0] # batch size
    D = data_mean.shape[0] # dimensionality
    orig_data = np.zeros((T, D), dtype=np.float32)
    dimensions_to_use = np.array([dim for dim in range(D)
                                    if dim not in dimensions_to_ignore])
    orig_data[:, dimensions_to_use] = normalized_data
    # multiply times stdev and add the mean
    stdMat = data_std.reshape((1, D))
    stdMat = np.repeat(stdMat, T, axis=0)
    meanMat = data_mean.reshape((1, D))
    meanMat = np.repeat(meanMat, T, axis=0)
    orig_data = np.multiply(orig_data, stdMat) + meanMat
    return orig_data

def define_actions(action):
    """
    Given an action string, returns a list of corresponding actions.

    Args
        action: String. either "all" or one of the h36m actions
    Returns
        actions: List of strings. Actions to use.
    Raises
        ValueError: if the action is not a valid action in Human 3.6M
    """
    actions = ["Directions",
               "Discussion",
               "Eating",
               "Greeting",
               "Phoning",
               "Photo",
               "Posing",
               "Purchases",
               "Sitting",
               "SittingDown",
               "Smoking",
               "Waiting",
               "WalkDog",
               "Walking",
               "WalkTogether"
               ]

    if action == "All" or action == "all":
        return actions

    if not action in actions:
        raise( ValueError, "Unrecognized action: %s" % action )

    return [action]

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
            sname = seqname[:-3] + "." + name + ".h5" # e.g.: Waiting 1.58860488.h5
            t2d[ (subj, a, sname) ] = pts2d
    return t2d
def project_to_cameras2(poses_set, cams, ncams=4):
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
    for t3dk in poses_set.keys():
        subj, a, seqname = t3dk
        t3d = poses_set[t3dk]
        for cam in range(ncams):
            R, T, f, c, k, p, name = cams[(subj, cam+1)]
            pts2d, _, _, _, _ = cameras.project_point_radial2(np.reshape(t3d, [-1, 3]), R, T, f, c, k, p)
            pts2d = np.reshape(pts2d, [-1, len(H36M_NAMES)*2])
            sname = seqname[:-3] + "." + name + ".h5" # e.g.: Waiting 1.58860488.h5
            t2d[ (subj, a, sname) ] = pts2d
    return t2d
def postprocess_3d(poses_set):
    """
    Center 3d points around root

    Args
        poses_set: dictionary with 3d data
    Returns
        poses_set: dictionary with 3d data centred around root (center hip) joint
        root_positions: dictionary with the original 3d position of each pose
    """
    root_positions = {}
    for k in poses_set.keys():
        # Keep track of the global position
        root_positions[k] = copy.deepcopy(poses_set[k][:,:3])
        # Remove the root from the 3d position
        poses = poses_set[k]
        poses = poses - np.tile( poses[:,:3], [1, len(H36M_NAMES)] )
        poses_set[k] = poses
    return poses_set, root_positions

def postprocess_2d(poses_set):
    """
    Center 2d points around root

    Args
        poses_set: dictionary with 2d data
    Returns
        poses_set: dictionary with 2d data centred around root (center hip) joint
        root_positions: dictionary with the original 2d position of each pose
    """
    root_positions = {}
    for k in poses_set.keys():
    # Keep track of the global position
        root_positions[k] = copy.deepcopy(poses_set[k][:,:2])
        # Remove the root from the 3d position
        poses = poses_set[k]
        poses = poses - np.tile( poses[:,:2], [1, len(H36M_NAMES)] )
        poses_set[k] = poses
    return poses_set, root_positions

def get_all_data(data_x,
                 data_y,
                 camera_frame,
                 norm_twoD=False,
                 input_size=32,
                 output_size=48
                 ):
    """
    Obtain numpy arrays for network inputs/outputs

    Args
      data_x: dictionary with 2d inputs
      data_y: dictionary with 3d expected outputs
      camera_frame: whether the 3d data is in camera coordinates
      input_size: input vector length for each sample
      output_size: output vector length for each sample
    Returns
      encoder_inputs: numpy array for the input data
      decoder_outputs: numpy array for the output data
    """
    # if norm_twoD:
    #     input_size -= 2
    # Figure out how many frames we have
    n = 0
    for key2d in data_x.keys():
      n2d, _ = data_x[ key2d ].shape
      n = n + n2d

    encoder_inputs  = np.zeros((n, input_size), dtype=np.float32)
    decoder_outputs = np.zeros((n, output_size), dtype=np.float32)

    # Put all the data into big arrays
    idx = 0
    for key2d in data_x.keys():
      (subj, b, fname) = key2d
      # keys should be the same if 3d is in camera coordinates
      key3d = key2d if (camera_frame) else (subj, b, '{0}.h5'.format(fname.split('.')[0]))
      # '-sh' suffix means detected key-points are used
      key3d = (subj, b, fname[:-3]) if fname.endswith('-sh') and camera_frame else key3d

      n2d, _ = data_x[ key2d ].shape
      encoder_inputs[idx:idx+n2d, :]  = data_x[ key2d ]
      decoder_outputs[idx:idx+n2d, :] = data_y[ key3d ]
      idx = idx + n2d

    return encoder_inputs, decoder_outputs

def prepare_dataset(opt):
    """
    Prepare PyTorch dataset objects used for training 2D-to-3D deep network

    Args
        opt: experiment options
    Returns
        train_dataset: training dataset as PyTorch dataset object
        eval_dataset: evaluation dataset as PyTorch dataset object
        data_stats: dataset statistics computed from the training dataset
        action_eval_list: a list of evaluation dataset objects where each
        corresponds to one action
    """
    # get relevant paths
    data_dir =  opt.data_dir
    cameras_path = os.path.join(data_dir, 'cameras.npy')
    # By default, all actions are used
    actions = define_actions(opt.actions)
    # load camera parameters to project 3D skeleton
    rcams = np.load(cameras_path, allow_pickle=True).item()
    # produce more camera views by adding virtual cameras if needed
    if opt.virtual_cams:
        rcams = add_virtual_cams(rcams)
    # first prepare Python dictionary containing 2D and 3D data
    data_dic, data_stats = prepare_data_dict(rcams,
                                             opt,
                                             predict_14=False
                                             )
    input_size = len(data_stats['dim_use_2d'])
    output_size = len(data_stats['dim_use_3d'])

    if opt.train:
        # convert Python dictionary to numpy array
        train_input, train_output = get_all_data(data_dic['train_set_2d'],
                                                 data_dic['train_set_3d'],
                                                 camera_frame,
                                                 norm_twoD=opt.norm_twoD,
                                                 input_size=input_size,
                                                 output_size=output_size
                                                 )
        # The Numpy arrays are finally used to initialize the dataset objects
        train_dataset = dataset.PoseDataset(train_input,
                                            train_output,
                                            'train',
                                            refine_3d = opt.refine_3d
                                            )
    else:
        train_dataset = None

    eval_input, eval_output = get_all_data(data_dic['test_set_2d'],
                                           data_dic['test_set_3d'],
                                           camera_frame,
                                           norm_twoD=opt.norm_twoD,
                                           input_size=input_size,
                                           output_size=output_size
                                           )

    eval_dataset = dataset.PoseDataset(eval_input,
                                       eval_output,
                                       'eval',
                                       refine_3d = opt.refine_3d
                                       )
    # Create a list of dataset objects for action-wise evaluation
    action_eval_list = split_action(data_dic['test_set_2d'],
                                    data_dic['test_set_3d'],
                                    actions,
                                    camera_frame,
                                    opt,
                                    input_size=input_size,
                                    output_size=output_size
                                    )

    return train_dataset, eval_dataset, data_stats, action_eval_list