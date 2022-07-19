"""
3D pose estimation based on 2D key-point coordinates as inputs.
Author: Shichao Li
Email: nicholas.li@connect.ust.hk
"""
# import sys
# sys.path.append("../")
import logging
import os
import copy
import torch
import numpy as np

import libs.parser.parse as parse
import libs.utils.utils as utils
import libs.dataset.h36m.data_utils as data_utils
import libs.trainer.trainer as trainer
import libs.dataset.h36m.data_utils as data_utils
import libs.model.modelegopose1 as model
from libs.utils.utils import compute_similarity_transform
import datetime
import torch.nn.functional as F
import torch
import numpy as np
import logging
from common.egopose_dataset import EgoposeDataset
import time
def normalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2

    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    return X / w * 2 - [1, h / w]
from torch.utils.data import  Dataset,DataLoader,TensorDataset
class trainset2(Dataset):
    def __init__(self, out_poses_3d, out_poses_2d, out_camera_rot, out_camera_trans):
        #定义好 image 的路径

        self.out_poses_3d = out_poses_3d
        self.out_poses_2d=out_poses_2d

        self.out_camera_rot=out_camera_rot
        self.out_camera_trans = out_camera_trans
    def __getitem__(self, index):

        out_poses_3d = self.out_poses_3d[index]
        out_poses_2d=self.out_poses_2d[index]

        out_camera_rot =self.out_camera_rot[index]
        out_camera_trans = self.out_camera_trans[index]
        return torch.from_numpy(out_poses_3d.astype(float) ),torch.from_numpy(out_poses_2d.astype(float) ),torch.from_numpy(out_camera_rot.astype(float) ),torch.from_numpy(out_camera_trans.astype(float) )

    def __len__(self):
        return len(self.out_poses_3d)
def main():
    # logging configuration
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s]: %(message)s"
                        )

    # parse command line input
    opt = parse.parse_arg()

    # Set GPU
    opt.cuda = opt.gpuid >= 0
    # if opt.cuda:
    #     torch.cuda.set_device(opt.gpuid)
    # else:
    #     logging.info("GPU is disabled.")

    # dataset preparation
    dataset_path= "/root/VideoPose3D/egopose_hm36_trainval/all_data.npz"
    dataset = EgoposeDataset(dataset_path)
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            anim = dataset[subject][action]

            # for idx, pos_3d in enumerate(anim['positions_3d']):
            #     #
            #     # pos_3d[0:10] -= pos_3d[10]  # Remove global offset, but keep trajectory in first position
            #     # pos_3d[11:] -= pos_3d[10]
            #     pos_3d = pos_3d/100
            #     # pos_3d[1:] -= pos_3d[:1]
            for idx, pos_2d in enumerate(anim['positions']):
                pos_2d = normalize_screen_coordinates(pos_2d, w=256, h=256)
                anim['positions'][idx] = pos_2d

    def fetch(subjects, subset=1):
        out_poses_3d = []
        out_poses_2d = []
        out_imag_path = []
        out_camera_params = []
        out_camera_rot = []
        out_camera_trans = []
        for subject in subjects:
            for env in dataset[subject].keys():
                poses_2d = dataset[subject][env]['positions']
                out_poses_2d.append(np.array(poses_2d))
                poses_3d = dataset[subject][env]['positions_3d']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                out_poses_3d.append(np.array(poses_3d))
                image_path = dataset[subject][env]['image_path']
                rot = dataset[subject][env]['rot']
                trans = dataset[subject][env]['trans']
                out_imag_path.append(np.array(image_path))
                out_camera_rot.append(np.array(rot))
                out_camera_trans.append(np.array(trans))
        return out_camera_params, out_poses_3d, out_poses_2d, out_imag_path, out_camera_rot, out_camera_trans

    _, poses_train, poses_train_2d, img_path_train, rot, trans = fetch(
        ['female_001_a_a', 'female_002_f_s', 'female_003_a_a', 'female_007_a_a', 'female_009_a_a', 'female_011_a_a',
         'female_014_a_a', 'female_015_a_a', 'male_003_f_s', 'male_004_a_a', 'male_005_a_a', 'male_006_f_s',
         'male_007_a_a', 'male_008_a_a', 'male_009_a_a', 'male_010_f_s', 'male_011_f_s', 'male_014_a_a'])
    _, test_poses_train, test_poses_train_2d, test_img_path_train, test_rot, test_trans = fetch(
        ["female_004_a_a", "female_008_a_a", "female_010_a_a", "female_012_a_a", "female_012_f_s", "male_001_a_a",
         "male_002_a_a", "female_004_a_a", "male_004_f_s", "male_006_a_a", "male_007_f_s", "male_010_a_a",
         "male_014_f_s"])
    poses_train = np.concatenate(poses_train, axis=0)
    poses_train_2d = np.concatenate(poses_train_2d, axis=0)
    test_poses_train = np.concatenate(test_poses_train, axis=0)
    test_poses_train_2d = np.concatenate(test_poses_train_2d, axis=0)

    rot = np.concatenate(rot, axis=0)
    trans = np.concatenate(trans, axis=0)
    deal_dataset2 = trainset2( out_poses_3d=poses_train, out_poses_2d=poses_train_2d,out_camera_rot=rot,out_camera_trans=trans)
    dataloader = DataLoader(dataset=deal_dataset2, batch_size=1024, shuffle=True)

    mymodel = model.get_model(
        stage_id=1,
        refine_3d=opt.refine_3d,
        norm_twoD=opt.norm_twoD,
        num_blocks=opt.num_blocks,
        input_size=34,
        output_size=51,
        linear_size=opt.linear_size,
        dropout=opt.dropout,
        leaky=opt.leaky)
    mymodel = mymodel.cuda()
    optim = torch.optim.Adam(mymodel.parameters(), lr=0.02, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.9)
    epoch=7000
    for batch_idx in range(1, epoch + 1):
        print("lr:{}".format(optim.state_dict()['param_groups'][0]['lr']))
        lossall = 0

        for i, (poses_train, poses_train_2d,rot, trans) in enumerate(dataloader):

            # poses_train = np.float(poses_train)
            # poses_train_2d = np.float(poses_train_2d)
            poses_train = poses_train.to(torch.float32)
            poses_train_2d = poses_train_2d.to(torch.float32)
            poses_train=poses_train.reshape(poses_train.shape[0],-1)
            poses_train_2d=poses_train_2d.reshape(poses_train_2d.shape[0],-1)
            if opt.cuda:
                with torch.no_grad():
                    # move to GPU
                    data_train = poses_train_2d.cuda()
                    data_targ = poses_train.cuda()
            optim.zero_grad()
            # forward pass to get prediction
            prediction = mymodel(data_train)
            loss = F.mse_loss(prediction, data_targ)
            loss.backward()
            lossall += loss
            optim.step()
        pred=prediction.reshape(prediction.shape[0],-1,3).cpu().detach().numpy()
        gt=data_targ.reshape(data_targ.shape[0],-1,3).cpu().detach().numpy()
        error = np.mean(np.sqrt(np.sum((pred*1000 - gt*1000)**2, axis=1)))
        print(error)
        scheduler.step()
        print("Epoch:{} Loss：{}".format(batch_idx, loss))
        if batch_idx%100==0:
            test_poses_train_2d=torch.tensor(test_poses_train_2d)
            test_poses_train_2d = test_poses_train_2d.to(torch.float32)
            test_poses_train_2d = test_poses_train_2d.reshape(test_poses_train_2d.shape[0], -1)
            test_poses_train_2d = test_poses_train_2d.cuda()
            prediction = mymodel(test_poses_train_2d)
            pred = prediction.reshape(prediction.shape[0], -1, 3).cpu().detach().numpy()
            error = np.mean(np.sqrt(np.sum((pred * 1000 - test_poses_train * 1000) ** 2, axis=1)))
            print("test==========================:{}".format(error))
        if batch_idx%500==0:
            time_str = time.strftime('%Y-%m-%d-%H-%M-%S')
            save2Dpath = os.path.join(opt.data_dir, time_str)
            if not os.path.exists(save2Dpath):
                os.mkdir(save2Dpath)
            save2Dpath = save2Dpath + "/" + "model"
            torch.save(mymodel, save2Dpath)
if __name__ == "__main__":
    main()