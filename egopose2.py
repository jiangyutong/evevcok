"""
3D pose estimation based on 2D key-point coordinates as inputs.
Author: Shichao Li
Email: nicholas.li@connect.ust.hk
"""
import time
import logging
import os
import sys
sys.path.append("../")
import copy
import torch
import numpy as np
from common.egopose_dataset import EgoposeDataset
import libs.parser.parseregopose as parse
import libs.utils.utils as utils
import libs.dataset.h36m.data_utils as data_utils
import libs.trainer.trainer as trainer
import libs.dataset.h36m.data_utils as data_utils
import libs.model.modelegopse as model
from libs.utils.utils import compute_similarity_transform
import common.evaluate as evaluate
import torch.nn.functional as F
import torch
import numpy as np
import logging
import libs.trainer.traineregopose as trainer
import libs.dataset.h36m.pth_datasetegopose as dataset
import common.evaluate as evaluation
import matplotlib.pyplot as plt
import transformations
def show3Dpose_hm36(vals, ax,imgType):
    ax.view_init(elev=15., azim=70)

    lcolor='b'
    rcolor='r'

    I = np.array( [0, 7, 8, 9, 8, 11, 12, 8, 14,  15, 0, 4, 5, 0,  1, 2])
    J = np.array( [7, 8, 9, 10,11, 12,13,14, 15, 16, 4, 5, 6, 1, 2, 3])

    LR = np.array([0, 0, 0, 0,
                    0, 0, 0,
                    1, 1, 1,
                    0, 0, 0,
                    1, 1, 1], dtype=bool)
    for i in np.arange( len(I) ):
        x, y, z = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
        ax.plot(x, y, z, lw=2, color = rcolor if LR[i] else lcolor)
    if imgType=="camera":
        ax.set_xlim3d([1, -1])
        ax.set_ylim3d([-1, 1])
        ax.set_zlim3d([2, 0])
        ax.set_aspect('auto')
    else:
        ax.set_xlim3d([2, -2])
        ax.set_ylim3d([-1, 2])
        ax.set_zlim3d([3, -1])
        ax.view_init(-69, 90)
    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white)
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)

    ax.tick_params('x', labelbottom = True)
    ax.tick_params('y', labelleft = True)
    ax.tick_params('z', labelleft = True)
def save_ckpt(opt, record):
    """
    Save training results.
    """
    cascade = record['cascade']
    if not opt.save:
        return False
    if opt.save_name is None:
        save_name = time.asctime()
        save_name += ('stages_' + str(opt.num_stages) + 'blocks_'
                      + str(opt.num_blocks) + opt.extra_str
                      )
    save_dir = os.path.join(opt.save_root, save_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print("路徑 ：：：：：",os.path.join(save_dir, 'model.th'))
    torch.save(cascade, os.path.join(save_dir, 'model.th'))
    print('Model saved at ' + save_dir)
    # return True
    return save_dir
def load_ckpt(opt,save_dir):
    opt.ckpt_path = save_dir
    print("os.path.join(opt.ckpt_dir, 'model.th')::::::",save_dir)
    cascade = torch.load(os.path.join(save_dir, 'model.th'))
    if opt.cuda:
        cascade.cuda()
    return cascade
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
        return out_poses_3d,out_poses_2d,out_camera_rot,out_camera_trans

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
    eval_body = evaluation.EvalBody()
    eval_upper = evaluation.EvalUpperBody()
    eval_lower = evaluation.EvalLowerBody()
    dataset_path= "/root/egopose_hm36_trainval/all_data.npz"
    dataset2 = EgoposeDataset(dataset_path)
    for subject in dataset2.subjects():
        for action in dataset2[subject].keys():
            anim = dataset2[subject][action]

            for idx, pos_3d in enumerate(anim['positions_3d']):
                #
                # pos_3d[0:10] -= pos_3d[10]  # Remove global offset, but keep trajectory in first position
                # pos_3d[11:] -= pos_3d[10]
                pos_3d -= pos_3d[0]
            # pos_3d[1:] -= pos_3d[:1]
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
            for env in dataset2[subject].keys():
                poses_2d = dataset2[subject][env]['positions']
                out_poses_2d.append(np.array(poses_2d))
                poses_3d = dataset2[subject][env]['positions_3d']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                out_poses_3d.append(np.array(poses_3d))
                image_path = dataset2[subject][env]['image_path']
                rot = dataset2[subject][env]['rot']
                trans = dataset2[subject][env]['trans']
                out_imag_path.append(np.array(image_path))
                out_camera_rot.append(np.array(rot))
                out_camera_trans.append(np.array(trans))
        return out_camera_params, out_poses_3d, out_poses_2d, out_imag_path, out_camera_rot, out_camera_trans

    _, poses_train, poses_train_2d, img_path_train, rot, trans = fetch(
        ['female_001_a_a', 'female_002_f_s', 'female_003_a_a', 'female_007_a_a', 'female_009_a_a', 'female_011_a_a',
         'female_014_a_a', 'female_015_a_a', 'male_003_f_s', 'male_004_a_a', 'male_005_a_a', 'male_006_f_s',
         'male_007_a_a', 'male_008_a_a', 'male_009_a_a', 'male_010_f_s', 'male_011_f_s', 'male_014_a_a'])
    # _, test_poses_train, test_poses_train_2d, test_img_path_train, test_rot, test_trans = fetch(
    #     ["female_004_a_a", "female_008_a_a", "female_010_a_a", "female_012_a_a", "female_012_f_s", "male_001_a_a",
    #      "male_002_a_a", "female_004_a_a", "male_004_f_s", "male_006_a_a", "male_007_f_s", "male_010_a_a",
    #      "male_014_f_s"])
    # _, test_poses_train, test_poses_train_2d, test_img_path_train, test_rot, test_trans = fetch(
    #     ["female_004_a_a", "female_008_a_a", "female_010_a_a", "female_012_a_a", "female_012_f_s", "male_001_a_a",
    #      "male_002_a_a", "female_004_a_a", "male_004_f_s", "male_006_a_a", "male_007_f_s", "male_010_a_a",
    #      "male_014_f_s"])
    _, test_poses_train, test_poses_train_2d, test_img_path_train, test_rot, test_trans = fetch(["male_008_a_a"]) 
    poses_train = np.concatenate(poses_train, axis=0)
    poses_train_2d = np.concatenate(poses_train_2d, axis=0)
    test_poses_train = np.concatenate(test_poses_train, axis=0)
    test_poses_train_2d = np.concatenate(test_poses_train_2d, axis=0)
    rot = np.concatenate(rot, axis=0)
    trans = np.concatenate(trans, axis=0)
    # deal_dataset2 = trainset2( out_poses_3d=poses_train, out_poses_2d=poses_train_2d,out_camera_rot=rot,out_camera_trans=trans)
    # dataloader = DataLoader(dataset=deal_dataset2, batch_size=1024, shuffle=True)
    # # test = trainset2(out_poses_3d=poses_train, out_poses_2d=poses_train_2d, out_camera_rot=rot,
    # #                           out_camera_trans=trans)
    # # test_dataloader = DataLoader(dataset=test, batch_size=1024, shuffle=True)
    train_dataset = dataset.PoseDataset(poses_train_2d.reshape(poses_train.shape[0],-1),
                                        poses_train.reshape(poses_train.shape[0],-1),
                                        )
    test_dataset = dataset.PoseDataset(test_poses_train_2d.reshape(test_poses_train_2d.shape[0], -1),
                                        test_poses_train.reshape(test_poses_train.shape[0], -1)
                                        )
    # record = trainer.train_cascade(train_dataset,
    #                                test_dataset,
    #                                opt
    #                                )
    # save_dir = save_ckpt(opt, record)
    cascade = load_ckpt(opt, "/root/model/Wed Jul 13 15:54:03 2022stages_3blocks_3")
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                                   batch_size=opt.batch_size,
                                                   shuffle=True,
                                                   num_workers=opt.num_threads
                                                   )
    N=0
    output_dir = "egopose_hm36_trainval/"
    os.makedirs(output_dir, exist_ok=True)
    for stage_id in range(len(cascade)):
        print("#"+ "="*60 + "#")
        logging.info("Model performance after stage {:d}".format(stage_id + 1))
        stage_model = cascade[2]
        dists=[]
        for batch_idx, batch in enumerate(test_loader):
            data = batch[0]
            target = batch[1]
            data = data.to(torch.float32)
            target = target.to(torch.float32)
            if opt.cuda:
                with torch.no_grad():
                    # move to GPU
                    data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
                    stage_model=stage_model.cuda()
            prediction = stage_model(data)
            # compute loss
            pred = prediction.reshape(prediction.shape[0], -1, 3).cpu().detach().numpy()
            gt = target.reshape(target.shape[0], -1, 3).cpu().detach().numpy()
            error = np.mean(np.sqrt(np.sum((pred * 1000 - gt * 1000) ** 2, axis=2)),axis=1)
            error = np.mean(error)
            print(error)
            dists.append(error)
            Mmaya = np.array([[1, 0, 0, 0],
                              [0, -1, 0, 0],
                              [0, 0, -1, 0],
                              [0, 0, 0, 1]])
            for i in range(len(pred)):
                N+=1
                if N>2000: break
                ax = plt.subplot(111, projection='3d')
                show3Dpose_hm36(pred[i], ax,imgType="camera")
                plt.savefig(output_dir + str(i).zfill(5) + '_3DHeadCentered.png', dpi=200, format='png', bbox_inches='tight')
                plt.close()
                translation = np.array(trans[i])
                rotation = np.array(rot[i]) * np.pi / 180.0
                Mf = transformations.euler_matrix(rotation[0],
                                                  rotation[1],
                                                  rotation[2],
                                                  'sxyz')
                Mf[0:3, 3] = translation
                Mf = np.linalg.inv(Mf)
                # M为相机的旋转和平移矩阵
                M = Mmaya.T.dot(Mf)
                # 从相机坐标系返回世界坐标系
                M_inv = np.linalg.inv(M[0:3, 0:3])
                ax = plt.subplot(111, projection='3d')
                predictionw = M_inv.dot((pred[i].T - M[0:3, 3:4]/100)).T
                show3Dpose_hm36(predictionw, ax,imgType="world")
                plt.savefig(output_dir + str(i).zfill(5) + '_3DHeadCenteredw.png', dpi=200, format='png', bbox_inches='tight')
                plt.close()
            eval_body.eval(pred, gt, None)
            eval_upper.eval(pred, gt, None)
            eval_lower.eval(pred, gt, None)
        res = {'FullBody': eval_body.get_results(),
                'UpperBody': eval_upper.get_results(),
                'LowerBody': eval_lower.get_results()}
        print(res)
        dists = np.vstack(dists)
        dists = dists.mean()
        print("dists::::::::::::::::::{}".format(dists))
if __name__ == "__main__":
    main()