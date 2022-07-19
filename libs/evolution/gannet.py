import torch.nn as nn
import numpy as np
import os
from torch.autograd import Variable
import libs.parser.parse as parse
import datetime
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import matplotlib.pyplot as plt
from torch.utils.data import  Dataset,DataLoader,TensorDataset
from libs.evolution.parameter import parse_arg
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
def cast_to_float(dic, dtype=np.float32):
    # cast to float 32 for space saving
    for key in dic.keys():
        dic[key] = dic[key].detach().cpu().numpy()
        dic[key] = dic[key].astype(dtype)
    return dic



# 定义判别器  #####Discriminator######使用多层网络来作为判别器
# 将图片28x28展开成784，然后通过多层感知器，中间经过斜率设置为0.2的LeakyReLU激活函数，
# 最后接sigmoid激活函数得到一个0到1之间的概率进行二分类。
class discriminator(nn.Module):
    def __init__(self):

        super(discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(96, 256),  # 输入特征数为784，输出为256
            nn.LeakyReLU(0.2),  # 进行非线性映射
            nn.Linear(256, 256),  # 进行一个线性映射
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()  # 也是一个激活函数，二分类问题中，
            # sigmoid可以班实数映射到【0,1】，作为概率值，
            # 多分类用softmax函数
        )

    def forward(self, x):
        x = self.dis(x)
        return x


# ###### 定义生成器 Generator #####
# 输入一个100维的0～1之间的高斯分布，然后通过第一层线性变换将其映射到256维,
# 然后通过LeakyReLU激活函数，接着进行一个线性变换，再经过一个LeakyReLU激活函数，
# 然后经过线性变换将其变成784维，最后经过Tanh激活函数是希望生成的假的图片数据分布
# 能够在-1～1之间。
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()

        self.gen = nn.Sequential(
            nn.Linear(96, 256),  # 用线性变换将输入映射到256维
            nn.ReLU(True),  # relu激活
            nn.Linear(256, 256),  # 线性变换
            nn.ReLU(True),  # relu激活
            nn.Linear(256, 96),  # 线性变换
            # nn.Tanh()  # Tanh激活使得生成数据分布在【-1,1】之间，因为输入的真实数据的经过transforms之后也是这个分布
        )

    def forward(self, x):
        x = self.gen(x)
        return x
class trainset(Dataset):
    def __init__(self, data):
        #定义好 image 的路径
        self.target = data
    def __getitem__(self, index):
        target = self.target[index]
        return target

    def __len__(self):
        return self.target.shape[0]

from libs.skeleton.anglelimits import \
to_local, to_global, get_skeleton, to_spherical, \
nt_parent_indices, nt_child_indices, \
is_valid_local, is_valid
nt_parent_indices = [13, 17, 18, 25, 26, 6, 7, 1, 2, 13]
nt_child_indices = [15, 18, 19, 26, 27, 7, 8, 2, 3, 14]
root = "../../resources/constraints"
template = np.load(os.path.join(root, 'template.npy'), allow_pickle=True).reshape(32,-1)
template_bones = template[nt_parent_indices, :] - template[nt_child_indices, :]
template_bone_lengths = to_spherical(template_bones)[:, 0]
nt_parent_indices = [13, 17, 18, 25, 26, 6, 7, 1, 2]
nt_child_indices = [15, 18, 19, 26, 27, 7, 8, 2, 3]
def re_order(skeleton):
    # the ordering of coordinate used by the Prior was x,z and y
    return skeleton[:, [0,2,1]]
def get_bone_length(skeleton):
    """
    Compute limb length for a given skeleton.
    """
    bones = skeleton[nt_parent_indices, :] - skeleton[nt_child_indices, :]
    bone_lengths = to_spherical(bones)[:, 0]
    return bone_lengths
def modify_pose(skeleton, local_bones, bone_length, ro=False):
    # get a new pose by modify an existing pose with input local bone vectors
    # and bone lengths
    new_bones = to_global(skeleton, local_bones)['bg']
    new_pose = get_skeleton(new_bones, skeleton, bone_length=bone_length)
    if ro:
        new_pose = re_order(new_pose)
    return new_pose.reshape(-1)
def set_z(pose, target):
    if pose is None:
        return None
    original_shape = pose.shape
    pose = pose.reshape(32, 3)
    min_val = pose[:, 2].min()
    pose[:, 2] -= min_val - target
    return pose.reshape(original_shape)

if __name__ == '__main__':
    post_processing = True
    # 创建对象
    D = torch.load('/media/zlz422/82BEDFE2BEDFCD33/jyt/project/EvSkeleton/data/human3.6M/two_gennet/D')
    G =  torch.load('/media/zlz422/82BEDFE2BEDFCD33/jyt/project/EvSkeleton/data/human3.6M/two_gennet/G')
    if torch.cuda.is_available():
        gpuid = 2
        torch.cuda.set_device(gpuid)
        D = D.cuda()
        G = G.cuda()
    epochs=500
    # 首先需要定义loss的度量方式  （二分类的交叉熵）
    # 其次定义 优化函数,优化函数的学习率为0.0003
    criterion = nn.BCELoss()  # 是单目标二分类交叉熵函数
    d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0003)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0003)
    opt = parse.parse_arg()  # 不是cvae
    scheduler_d = torch.optim.lr_scheduler.MultiStepLR(d_optimizer,
                                                     milestones=[1000,1500], gamma=0.9)
    scheduler_g = torch.optim.lr_scheduler.MultiStepLR(g_optimizer,
                                                     milestones=[1000,1500], gamma=0.9)
    dict_path = os.path.join(opt.data_dir_cvae, 'threeDPose_train.npy')
    train_dict_3d = np.load(dict_path, allow_pickle=True).item()
    dict_path = os.path.join(opt.data_dir_cvae, 'threeDPose_test.npy')
    test_dict_3d = np.load(dict_path, allow_pickle=True).item()
    mydatarget=[]
    num_dataThree=[]
    for key in train_dict_3d.keys():
        targets = train_dict_3d[key]
        targets = torch.Tensor(targets)
        num_dataThree.append(targets.shape[0])
        mydatarget.append(targets)
    mydatarget = np.concatenate(mydatarget,axis=0)

    deal_dataset=trainset(mydatarget)
    real_data = DataLoader(dataset=deal_dataset,batch_size=256,shuffle=True)
    # optim = torch.optim.Adam(mymodel.parameters(), lr=1, weight_decay=0.0)
    # inputs_get = {}
    # for key in train_dict_3d.keys():
    #     targets = train_dict_3d[key]
    #     mymean = np.mean(targets, axis=0)
    #     mymean = np.mean(mymean, axis=0)
    #     mystd = np.std(targets, axis=0)
    #     mystd = np.std(mystd, axis=0)
    #     # 7、创建一个3x3的，均值为0，方差为1,正太分布的随即数数组
    #     inputs = np.random.normal(mymean, mystd, targets.shape)
    #     inputs_get[key] = inputs
    finaldata=torch.zeros(size=mydatarget.shape)
    offsprings_r_all = []
    # ##########################进入训练##判别器的判断过程#####################
    for epoch in range(1,epochs+1):  # 进行多个epoch的训练
        scheduler_d.step()
        scheduler_g.step()
        print("lr:{}".format(d_optimizer.state_dict()['param_groups'][0]['lr']))
        print("lr:{}".format(g_optimizer.state_dict()['param_groups'][0]['lr']))
        j=0
        for i,targets in enumerate(real_data):
            if epoch == epochs:
                # 保存模型
                now = datetime.datetime.now()
                stpath = str(now) + "_" + "gennet"
                save2Dpath = os.path.join(opt.data_dir_cvae, stpath)

                z = Variable(torch.randn(targets.shape)).cuda()
                fake_data = G(z)  # 随机噪声输入到生成器中，得到一副假的图片
                finaldata[j:j+256]=fake_data
                j=j+256
                continue
            targets = torch.Tensor(targets)
            num_img = targets.shape[0]
            real_label = torch.ones(num_img,1)
            fake_label = torch.zeros(num_img,1)
            targets = Variable(targets.cuda())
            real_label = Variable(real_label).cuda()  # 定义真实的图片label为1
            fake_label = Variable(fake_label).cuda()  # 定义假的图片的label为0
            # ########判别器训练train#####################
            # 分为两部分：1、真的图像判别为真；2、假的图像判别为假
            # 计算真实图片的损失
            real_out = D(targets)  # 将真实图片放入判别器中
            d_loss_real = criterion(real_out, real_label)  # 得到真实图片的loss
            real_scores = real_out  # 得到真实图片的判别值，输出的值越接近1越好
            # 计算假的图片的损失
            z = Variable(torch.randn(targets.shape)).cuda()
            fake_data = G(z).detach()  # 随机噪声放入生成网络中，生成一张假的图片。 # 避免梯度传到G，因为G不用更新, detach分离
            fake_out = D(fake_data)  # 判别器判断假的图片，
            d_loss_fake = criterion(fake_out, fake_label)  # 得到假的图片的loss
            fake_scores = fake_out  # 得到假图片的判别值，对于判别器来说，假图片的损失越接近0越好
            # 损失函数和优化
            d_loss = d_loss_real + d_loss_fake  # 损失包括判真损失和判假损失
            d_optimizer.zero_grad()  # 在反向传播之前，先将梯度归0
            d_loss.backward()  # 将误差反向传播
            d_optimizer.step()  # 更新参数

            # ==================训练生成器============================
            # ###############################生成网络的训练###############################
            # 原理：目的是希望生成的假的图片被判别器判断为真的图片，
            # 在此过程中，将判别器固定，将假的图片传入判别器的结果与真实的label对应，
            # 反向传播更新的参数是生成网络里面的参数，
            # 这样可以通过更新生成网络里面的参数，来训练网络，使得生成的图片让判别器以为是真的
            # 这样就达到了对抗的目的
            # 计算假的图片的损失
            z = Variable(torch.randn(targets.shape)).cuda()
            fake_data = G(z)  # 随机噪声输入到生成器中，得到一副假的图片
            output = D(fake_data)  # 经过判别器得到的结果
            g_loss = criterion(output, real_label)  # 得到的假的图片与真实的图片的label的loss
            # bp and optimize
            g_optimizer.zero_grad()  # 梯度归0
            g_loss.backward()  # 进行反向传播
            g_optimizer.step()  # .step()一般用在反向传播后面,用于更新生成网络的参数

            # 打印中间的损失
            if (i + 1) % 1500 == 0:
                print('Epoch[{}/{}],d_loss:{:.6f},g_loss:{:.6f} '
                      'D real: {:.6f},D fake: {:.6f}'.format(
                    epoch, epochs, d_loss.data.item(), g_loss.data.item(),
                    real_scores.data.mean(), fake_scores.data.mean()  # 打印的是真实图片的损失均值
                ))
            offsprings_r = fake_data.detach().cpu().numpy()
            # np.save("/media/zlz422/82BEDFE2BEDFCD33/jyt/project/EvSkeleton/data/jyt.npy",offsprings)
            tmpLeng = len(offsprings_r)
            tmpi = 0
            for m in range(tmpLeng):
                offsprings_r_tmpi = offsprings_r[tmpi]
                offsprings_r_tmpi = offsprings_r_tmpi.reshape(-1, 3)
                offsprings_r_tmpi = offsprings_r_tmpi
                offsprings_r_tmpi = re_order(offsprings_r_tmpi)
                offsprings_r_tmpi_length = get_bone_length(offsprings_r_tmpi)
                bones_offsprings_r_tmpi = to_local(offsprings_r_tmpi)
                    # apply joint angle constraint as the fitness function
                valid_vec_fa = is_valid_local(bones_offsprings_r_tmpi)
                if valid_vec_fa.sum() >= 9:
                    offsprings_r_tmpi = modify_pose(offsprings_r_tmpi, bones_offsprings_r_tmpi,
                                                    offsprings_r_tmpi_length,
                                                    ro=True)
                    if post_processing:
                        # move the poses to the ground plane
                        set_z(offsprings_r_tmpi, np.random.normal(loc=20.0, scale=3.0))
                    offsprings_r[tmpi] = offsprings_r_tmpi
                    offsprings_r_all.append(offsprings_r_tmpi)
                    # ax1 = plt.subplot(1, 1, 1, projection='3d')
                    # ax1.grid(False)
                    # plt.axis('off')
                    # # plt.title('father')
                    # show3Dpose(offsprings_r_tmpi, ax1, add_labels=False, plot_dot=True)
                    # plt.tight_layout()
                else:
                    offsprings_r = np.delete(offsprings_r, tmpi, 0)
                    tmpi = tmpi - 1
                    tmpLeng = tmpLeng - 1
                tmpi = tmpi + 1
            if epoch%2==0:
                stpath = "two" + "_" + "gennet"
                save2Dpath = os.path.join(opt.data_dir_cvae, stpath)
                if not os.path.exists(save2Dpath):
                    os.mkdir(save2Dpath)
                saveGpath = save2Dpath + "/" + "G"
                saveDpath = save2Dpath + "/" + "D"
                torch.save(G, saveGpath)
                torch.save(D, saveDpath)
                finaldatakey_vail = {}
                finaldatakey_vail[('1', 'n/a', 'n/a')] = offsprings_r_all
                for key in finaldatakey_vail.keys():
                    finaldata_vail = np.zeros((len(finaldatakey_vail[key]), 96), dtype=float)
                    for i in range(len(finaldatakey_vail[key])):
                        finaldata_vail[i] = finaldatakey_vail[key][i]
                    finaldatakey_vail[key] = finaldata_vail
                save_path = "/media/zlz422/82BEDFE2BEDFCD33/jyt/project/EvSkeleton/data/human3.6M/jyt_gennet_random_vailm"+str(epoch)
                np.save(save_path, finaldatakey_vail)
                a=np.load(save_path+".npy",allow_pickle=True).item()
                print("save:{}".format(save_path))

