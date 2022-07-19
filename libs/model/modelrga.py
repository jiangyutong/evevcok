"""
Fully-connected residual network as a single deep learner.
"""
import torch.nn as nn
import math

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

import torch.nn as nn
import torch
from libs.model.rga import RGA_Module



class ResidualBlock(nn.Module):
    """
    A residual block.
    """

    def __init__(self, linear_size, p_dropout=0.5, kaiming=True, leaky=False):
        super(ResidualBlock, self).__init__()
        self.l_size = linear_size
        if leaky:
            self.relu = nn.LeakyReLU(inplace=True)
        else:
            self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)

        self.w2 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)
        if kaiming:
            self.w1.weight.data = nn.init.kaiming_normal_(self.w1.weight.data)
            self.w2.weight.data = nn.init.kaiming_normal_(self.w2.weight.data)
        # # 网络的第一层加入注意力机制
        # self.ca = ChannelAttention(self.l_size)
        # self.sa = SpatialAttention()
        # # 网络的卷积层的最后一层加入注意力机制
        # self.ca1 = ChannelAttention(self.l_size)
        # self.sa1 = SpatialAttention()

    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        return out


class FCModel(nn.Module):
    def __init__(self,
                 linear_size=1024,
                 num_blocks=2,
                 p_dropout=0.5,
                 norm_twoD=False,
                 kaiming=False,
                 refine_3d=False,
                 leaky=False,
                 dm=False,
                 input_size=64,
                 output_size=64):
        """
        Fully-connected network.
        """
        super(FCModel, self).__init__()

        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_blocks = num_blocks
        self.refine_3d = refine_3d
        self.leaky = leaky
        self.dm = dm
        self.input_size = input_size
        self.output_size = output_size

        # process input to linear size
        # self.con=nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1)
        self.w11 = nn.Linear(self.input_size, self.input_size)
        self.w12 = nn.Linear(self.input_size, self.output_size)
        self.batch_norm11 = nn.BatchNorm1d(self.input_size)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)
        # self.res_blocks=ResidualBlock(self.linear_size,
        #                                      self.p_dropout,
        #                                      leaky=self.leaky)
        # output
        # self.w2 = nn.Linear(self.linear_size, self.output_size)
        if self.leaky:
            self.relu = nn.LeakyReLU(inplace=True)
        else:
            self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

        if kaiming:
            self.w11.weight.data = nn.init.kaiming_normal_(self.w11.weight.data)
            self.w12.weight.data = nn.init.kaiming_normal_(self.w12.weight.data)

        self.rga = RGA_Module(in_channel=32, size=20, use_spatial=True, use_channel=True)
    def forward(self, x):
        # x1=x.reshape(-1,32,2)
        # x1 = self.con(x1)
        # x1 = x1.reshape(-1,64)
        # y=x+x1
        y = self.w11(x)
        y = self.batch_norm11(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = y + x
        y = self.w12(y)
        y2 = x.reshape(-1, 32, 2)
        y2 = y2.transpose(1, 2)
        mysize = y2.shape[0]
        y = self.rga(y2)


        return y

    # def get_representation(self, x):
    #     # get the latent representation of an input vector
    #     # first layer
    #     # x = self.ca(x) * x
    #     # x=x.reshape(-1,32,2)
    #     # x=x.transpose(1, 2)
    #     # y = self.lstm(x)
    #     #
    #     # y = y[0].reshape(-1,64)
    #     y=self.w11(x)
    #     y = self.batch_norm11(y)
    #     y = self.relu(y)
    #     y = self.dropout(y)
    #     # y = self.w11(y)
    #     # y = self.batch_norm11(y)
    #     # y = self.relu(y)
    #     # y = self.dropout(y)
    #     y=y+x
    #     # y = self.w1(y)
    #     # y = self.batch_norm1(y)
    #     # y = self.relu(y)
    #
    #     # residual blocks
    #     # y = self.res_blocks(y)
    #     y=self.w12(y)
    #     return y


def get_model(
              refine_3d=False,
              norm_twoD=False,
              num_blocks=2,
              input_size=64,
              output_size=64,
              linear_size=1024,
              dropout=0.5,
              leaky=False
              ):
    model = FCModel(
                    refine_3d=refine_3d,
                    norm_twoD=norm_twoD,
                    num_blocks=num_blocks,
                    input_size=input_size,
                    output_size=output_size,
                    linear_size=linear_size,
                    p_dropout=dropout,
                    leaky=leaky
                    )
    return model


def prepare_optim(model, opt):
    """
    Prepare optimizer.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    if opt.optim_type == 'adam':
        optimizer = torch.optim.Adam(params,
                                     lr=opt.lr,
                                     weight_decay=opt.weight_decay
                                     )
    elif opt.optim_type == 'sgd':
        optimizer = torch.optim.SGD(params,
                                    lr=opt.lr,
                                    momentum=opt.momentum,
                                    weight_decay=opt.weight_decay
                                    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=opt.milestones,
                                                     gamma=opt.gamma)
    return optimizer, scheduler


def get_cascade():
    """
    Get an empty cascade.
    """
    return nn.ModuleList([])